# Block calibration and teacher reconstruction

`BlockQuantizationPipeline` runs existing recipes on a lazy sequence of independent
float blocks. It carries aligned reference and quantized-upstream activation
streams, collects statistics on a forward device, and estimates/packs weights on
an independently selected device. A block stays on the forward device from its
reference forward to its student forward, except where a stage needs it
elsewhere: estimation moves it to the estimation device, and an
`inference_model` view is built from a host copy. With distinct forward and
estimation devices, or with a view, it therefore crosses devices once per stage.
It returns to the host at the end, also when a stage fails. The caller supplies block
boundaries and any batch propagation convention; the runner has no GPTQ or
model-family dispatch.

There are three useful configurations:

| Statistics input | GPTQ `reconstruction` | Local objective |
| --- | --- | --- |
| `input_source="float"` | `"self"` | Preserve the layer on original float activations |
| `input_source="sequential"` | `"self"` | Preserve the layer on quantized-upstream activations |
| `input_source="sequential"` | `"teacher"` | Match the original float linear output using student inputs |

The float-input schedule matches whole-model float statistics for a deterministic
partition. The sequential schedule does not inherently correct upstream error.
Teacher reconstruction adds a ridge-regularized weight correction before GPTQ:
with `H` the input Hessian and `C` the teacher-minus-student output/input product,
`W_fit = W + C (H + lambda I)^-1`. Lambda is
`reconstruction_damp_percent * mean(diag(H))`, defaulting to 0.01 of that mean.
Bias remains fixed. GPTQ then quantizes `W_fit` using the same input Hessian.

Teacher reconstruction currently supports `nn.Linear`. It is local reconstruction
of each linear invocation, not optimization of a whole nonlinear decoder block.
All statistics within a block are collected before its weights are adapted.
Quality relative to float-input or sequential GPTQ must be measured on held-out
samples; the teacher objective does not guarantee a better final model.
`"teacher"` needs `reference_data`, the teacher's input stream: the block runner
supplies it, a whole-model `QuantizationPipeline.run` must be given it and
otherwise fails before calibration.

```python
from neuropress import BlockQuantizationPipeline, QuantizationRecipeConfig

config = QuantizationRecipeConfig.model_validate({
    "estimator": {
        "type": "gptq", "target": "linear", "nbits": 4,
        "symmetric": True, "group_size": 32, "weight_transform": "q4_0",
        "reconstruction": "teacher", "reconstruction_damp_percent": 0.01,
    },
    "kernel_lowering": None,
})
runner = BlockQuantizationPipeline(
    config,
    input_source="sequential",
    forward_device="cuda",
    estimation_device="cpu",
    checkpoint_dir="checkpoints",
    run_id="model-revision/dataset-selection/forward-contract",
)
for result in runner.run(named_float_blocks, calibration_batches):
    # result.model is the adapted block; result.kept_float records any fallback.
    consume(result)
```

By default, each block output becomes the next block input (a dict output is
passed as keyword arguments). Supply `propagate(previous_batch, output)` to
preserve masks or other arguments. Streams are plain CPU tensor trees so that
checkpoints reload with `weights_only=True`: a Hugging Face `ModelOutput` arrives
as a `dict` and a namedtuple as a `tuple`, in fresh and resumed runs alike, so
read `output["last_hidden_state"]` rather than the attribute. Use
`run_block` directly when the model needs its own traversal. Blocks must not share
weights across boundaries. Apply whole-model preprocessing before partitioning,
and kernel lowering after calibration. An optional `inference_model` callback can
provide a non-mutating inference view, for example f32 materialization of packed
weights for GPU calibration; include that contract in `run_id`. The view serves
every statistics collection and the propagated student stream, with or without
`target_paths`, so it must also accept the untouched float block. With
`target_paths`, the first stage pairs against that view of the still-float
block, while later stages pair against a frozen copy of the float block that
does not go through the view; a view that changes numerics (f32 materialization
of bf16 weights, say) gives the first stage a slightly different teacher.

Checkpoints store the adapted model and both propagated streams. A completed
checkpoint is reused only when the float weights, incoming streams, recipe,
device types, torch and neuropress versions and run identity match (`cuda` and
`cuda:0` count as the same). The package version changes only at release, so
when running from a development checkout put the code revision in `run_id`.
Each block keeps one checkpoint slot per fingerprint, so a run that changes any
of these computes into a new slot and never reuses stale data, while the earlier
slot stays valid for its own run. Missing completion metadata triggers
recomputation. With checkpointing
enabled, stream leaves must be tensors or plain scalars (`None`, numbers,
strings, dtypes, devices); any other object raises a `TypeError` up front,
since its fingerprint would differ between runs and it could not be reloaded.
Old slots are never pruned; delete the directory to reclaim space. Concurrent
writers are unsupported.

Paired observations are a collector capability, not a GPTQ runner branch.
Register a `PairedCollector` in `COLLECTOR_REGISTRY` and declare its statistic in
an estimator to receive aligned student inputs/output and teacher output.
Ordinary and paired requests share the same collection pass, including in
`CompositeEstimator` recipes. `QuantizationPipeline.run` also accepts
`reference_data`, `reference_model` and `estimation_device` directly. Teacher
outputs wait for the student pass on `reference_device`: the host by default,
which bounds device memory when a whole model pairs every linear. Block runs pass
`None` to keep one block's teacher outputs on the forward device, which saves a
round trip per paired output and batch. Nested composite configs preserve their
full estimator-specific JSON fields.
Use `target_key=""` to collect paired observations at the block root for a
whole-block estimator. Collection and stream propagation run without gradients;
estimation can use autograd, including differentiation through nonlinear blocks.
Collections without paired statistics run under `torch.inference_mode()`, so
their statistics are inference tensors; paired collection uses `torch.no_grad()`
so that paired samples stay usable in autograd.

Collectors receive a module's call arguments in declaration order, with keyword
arguments placed where the parameter is declared and `*args`/`**kwargs`
expanded. An unpaired collector takes the first of them, so a module called only
with keywords (`m(hidden_states=x)`) is observed on its input, like `m(x)`.
Earlier versions fell back to the output for keyword-only calls, so activation
statistics of such modules change.

## Dependency stages within a block

Set `target_paths` on `BlockQuantizationPipeline` to alternate collection and
adaptation at dependency barriers inside each block. The runner traces the
explicit target modules across the supplied calibration batches, groups targets
with no observed dependency, then runs a fresh recipe for each group. Later
stages collect new statistics through the already adapted stages. Teacher
observations always see the unchanged float block: the first stage collects
before any adaptation and pairs against the observed block itself, and later
stages use a frozen copy kept in host memory. That copy is made only if a later
stage's estimator requests paired statistics, and placed on the forward device
only for those stages, so self-reconstruction recipes never hold a second block
in host or device memory.

Use `target_paths` with `reconstruction="teacher"` whenever a block holds
dependent targets. Without stages, every target in the block is fitted to the
teacher from inputs collected before any of them is adapted, so each one
corrects the whole upstream error on its own; the corrections compound, and the
block can end up further from the teacher than with `"self"`. Staged
recollection lets each stage fit through the already adapted ones.

```python
from neuropress import QuantizationSchedule

paths = tuple(name for name, module in block.named_modules()
              if isinstance(module, torch.nn.Linear))
schedule = QuantizationSchedule.trace(block.eval(), calibration_batches, paths)
print(schedule)
runner = BlockQuantizationPipeline(config, target_paths=paths,
                                   input_source="sequential")
result = runner.run_block("block", block, calibration_batches, reference_batches)
print(result.schedule)
```

For a Qwen decoder block, the notation is:

```text
[q_proj | k_proj | v_proj] → [o_proj] → [gate_proj | up_proj] → [down_proj]
```

`|` means targets can share a statistics collection pass; `→` means apply the
preceding stage before recollecting. This expresses potential concurrency, not
parallel execution. Names above omit the attention/MLP prefixes for readability.
The ranking follows tensor dependencies through functional operations, reshapes,
residual additions and parameter reads, rather than Python hook call order.
Targets may be embeddings, convolutions or other modules supported by the recipe.

The trace describes observed eager PyTorch tensor dataflow, not arbitrary program
semantics. Supply representative batches and use a pure, deterministic eval
forward. In-place tensor operations, tensor-dependent Python branches, repeated
calls to a target, inputs depending on the target's own parameters, nested targets,
shared target parameter storage and targets
missing from a batch are rejected. Python side effects, foreign computation and
unobserved paths are outside this contract. Every selected target must be handled
by the recipe; assignments outside the selected weight stage are rejected. A
target the estimator tolerates failing on (such as a GPTQ Cholesky failure) counts
as handled: it stays float and is reported in `kept_float`, as in whole-block mode.

Batches with the same structure (tree, shapes, strides, dtypes, devices and scalar
leaves) replay the same operations, because tensor-dependent branches are
rejected, so the trace runs once per distinct structure. A fixed-length
calibration set costs one traced forward per block. Forwards that produce
value-dependent shapes (`nonzero`, boolean masks, token routing) are traced on
every batch.

Existing estimators with `filter_fn` support stage selection while preserving
format filters. Composite estimators forward selection to each member. Future
estimators can implement `select_targets(paths)` as a context manager, restoring
their selection state when the scope exits, including on failure. Global preprocessing
and final kernel lowering remain outside the staged traversal; methods requiring
joint optimization across dependent targets should use the default whole-block
mode. An `inference_model` callback can materialize packed weights for subsequent
GPU collection without changing the stored model. Stage settings are part of the
checkpoint fingerprint, and resumed results retain their schedule.
