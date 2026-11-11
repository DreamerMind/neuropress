# Block calibration and teacher reconstruction

`BlockQuantizationPipeline` runs existing recipes on a lazy sequence of independent
float blocks. It carries aligned reference and quantized-upstream activation
streams, collects statistics on a forward device, and estimates/packs weights on
an independently selected device. The caller supplies block boundaries and any
batch propagation convention; the runner has no GPTQ or model-family dispatch.

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
`target_paths`, so it must also accept the untouched float block.

Checkpoints store the adapted model and both propagated streams. A completed
checkpoint is reused only when the float weights, incoming streams, recipe,
devices and run identity match. Missing completion metadata triggers recomputation;
a mismatched completed checkpoint raises instead of silently reusing stale data.
Use separate directories for independent runs; concurrent writers are unsupported.

Paired observations are a collector capability, not a GPTQ runner branch.
Register a `PairedCollector` in `COLLECTOR_REGISTRY` and declare its statistic in
an estimator to receive aligned student inputs/output and teacher output.
Ordinary and paired requests share the same collection pass, including in
`CompositeEstimator` recipes. `QuantizationPipeline.run` also accepts
`reference_data`, `reference_model` and `estimation_device` directly. Nested
composite configs preserve their full estimator-specific JSON fields.
Use `target_key=""` to collect paired observations at the block root for a
whole-block estimator. Collection and stream propagation run without gradients;
estimation can use autograd, including differentiation through nonlinear blocks.

## Dependency stages within a block

Set `target_paths` on `BlockQuantizationPipeline` to alternate collection and
adaptation at dependency barriers inside each block. The runner traces the
explicit target modules across the supplied calibration batches, groups targets
with no observed dependency, then runs a fresh recipe for each group. Later
stages collect new statistics through the already adapted stages. Teacher
observations always see the unchanged float block: the first stage collects
before any adaptation and pairs against the observed block itself, and later
stages use a frozen copy kept in host memory. That copy is placed on the forward
device only for stages whose estimator requests paired statistics, so
self-reconstruction recipes never hold a second block on the device.

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
