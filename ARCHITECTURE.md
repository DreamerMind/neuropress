# neuropress architecture

## Commands

```bash
# Install dependencies (Python 3.13, managed with uv)
uv sync

# Install with optional LLM data loader (HuggingFace datasets)
uv sync --extra llm

# Run all unit tests
uv run pytest

# Run a single test file
uv run pytest tests/test_pipeline.py -v

# Run a single test by name
uv run pytest tests/test_pipeline.py::TestQuantizationPipeline::test_pipeline_runs -v

# Run integration tests (downloads ~500 MB model, requires RUN_INTEGRATION=1)
RUN_INTEGRATION=1 uv run pytest tests/integration/ -v
```

## Architecture

`neuropress` is a graph-aware neural network quantization framework targeting NNEF export via `torch_to_nnef`. The full pipeline proceeds in six steps (see `pipeline.py`):

1. **QGraph** – `RuntimeModuleGraphProvider` traces a single forward pass using PyTorch module hooks to build a `QGraph` (`nodes: Dict[str, QNode]`, `values: Dict[str, QValue]`). No symbolic tracing (FX/Dynamo) is used. The pipeline traces only when the estimator sets `requires_graph` (the base-class default); the built-in estimators do not read the graph and receive an empty one.

2. **Targets** – `WeightTarget` (`module_id/param_name`) and `ActivationTarget` (`value_id`) identify what to quantize.

3. **StatisticsRegistry** – `QuantizationParamEstimator.declare_statistics()` returns `StatisticRequest` objects. The registry deduplicates them (key = `target_key::stat_type`), installs hooks, runs a forward pass over calibration data, and stores results. Supported stat types include `"min_max"`, `"hessian"` and the paired `"linear_reconstruction"`. A `PairedCollector` also receives the matching output of a teacher model run on aligned `reference_data`; the pipeline builds a teacher only if `requires_pairing()` says some request needs one. With an `estimation_device`, `registry.to()` makes `get()` return each statistic on that device at access time, one at a time, while `inject()` hands statistics over as collected (GPTQ places each Hessian itself). After `prepare()`, `declare_targets()` lists the module paths the estimator quantizes, which statistics cannot tell (min-max and HQQ request none); the base reads `_target_modules` as filled by `collect_target_modules`, and estimators keeping targets elsewhere override it. `prepare()` runs once, before collection; moving to the estimation device does not rebind the estimator, so `estimate()` reads device placement from its target modules. `collection_model` lets statistics come from an inference view of the model, and `target_paths` scopes the estimator through `select_targets()` and rejects assignments outside that set.

4. **QuantizationRecipe** – ties together `allowed_weight_qschemes`, `allowed_activation_qschemes`, a `QuantizationParamEstimator`, and a `ModuleQuantizationAdapter`.

5. **QuantizationPlan** – the output of estimation: dicts mapping target keys to selected values. A weight assignment is a `QuantizedWeight` (integer codes, a `ZpScaleScheme` or `QSchemeGroup`, and an optional destination format) or an already-materialized tensor such as a VQ `QTensor`. Scalar estimators leave model parameters untouched; tied modules selected as targets share one assignment object. Codes wait on the CPU until the adapter runs, as they take one byte per weight. `kept_float` records selected layers that stayed float and why; size estimates come from the recipe alone and do not reflect them.

6. **ModuleQuantizationAdapter** – applies the plan to the model. `WeightAssignmentAdapter` converts each `QuantizedWeight` to its destination format without recalibrating, replaces mixed-precision `nn.Linear` modules with `MixedPrecisionLinear`, and keeps tied modules tied. A tied module listed with its own assignment, such as an embedding and an `lm_head` in different formats, receives its own tensor. A weight whose scale or offset is outside the format's f16 range stays float, so one layer cannot discard the others. Mixed branches with no tract format export to NNEF as float variables; standalone `q4_1_packed` and `q2_0_packed` tensors refuse NNEF export. A weight its destination cannot store, including a packing that could not restore its shape, stays float and is added to `kept_float` rather than discarding the other layers; blocks whose scale is below the f16 range are stored as exact zeros.

### Key abstractions

| Class | Role |
|---|---|
| `QSchemeDescriptor` | Static backend constraint (nbits, symmetric, granularity) |
| `ZpScaleScheme` | Concrete affine scheme (zero_point, scale, nbits); extends `torch_to_nnef.QScheme` |
| `QSchemeGroup` / `PanelSpec` / `TensorSlice` | Mixed-precision grouping of weight regions |
| `CalibrationMethod` | ABC for algorithms; subclasses implement `update_statistics` + `compute_qscheme` |
| `CalibrationMethodOnActivation` | Subclass of above for Hessian-based methods (GPTQ) |
| `GPTQCalibration` | Full GPTQ implementation (Frantar et al. ICLR 2023) |
| `GPTQMixedPrecisionCalibration` | GPTQ with per-column-group bit-width allocation via `MixedAllocationStrategy` |
| `MixedPrecisionLinear` | Drop-in replacement for `nn.Linear` produced by `split_mixed_precision_linear` |
| `PairedCollector` | Collector fed aligned student inputs/output and teacher output (`update_pair`) |
| `BlockQuantizationPipeline` | Runs a recipe one block at a time, propagating aligned float and student streams, with optional checkpoint resume |
| `QuantizationSchedule` | Dependency stages of target modules, traced from tensor dataflow, for collect/adapt barriers inside a block |

### Calibration data shapes (4-D convention)

Weights are reshaped to `[rows, 1, n_groups, group_size]` before quantization. Odd dims (1, group_size) are "reduction" dims; even dims (rows, n_groups) are "kept". `qscheme_data_shape` is the same rank with reduction dims set to 1.

### `StatField` annotation pattern

`CalibrationMethod` subclasses declare statistic dependencies via `Annotated` type hints. The registry auto-discovers and injects them — no manual wiring needed.

```python
hessian: Annotated[Optional[torch.Tensor], StatField("hessian", "hessian")] = None
```

`StatField(stat_type, field_key)`:
- `stat_type` — matches the string key in `StatisticRequest` (`"hessian"`, `"hessian_diag"`, `"min_max"`).
- `field_key` — the attribute name to populate on the method instance.

Flow: `StatisticsRegistry.register_method(method, target_key)` scans `Annotated` hints to build `StatisticRequest` objects. After collection, `registry.inject(method, target_key)` sets the annotated fields from stored results. The pattern lives across three files: `calibration/probe/collectors.py` (StatField), `calibration/probe/registry.py` (discover + inject), and each `CalibrationMethod` subclass.

### Estimator config / implementation split

`VptqGroupEstimatorConfig` (`estimator_configs.py`) is the user-facing Pydantic model; `VPTQGroupEstimator` (`estimators.py`) is the runtime dataclass. The config's `build()` passes every field explicitly, so **`VPTQGroupEstimator` is the single source of truth for defaults** — `VptqGroupEstimatorConfig` derives its defaults from the dataclass fields via `_vd()` at class definition time. If you add a field to `VPTQGroupEstimator`, add a corresponding `_vd("field_name")` default in the config.

### Block calibration (`block_pipeline.py`, `graph/schedule.py`)

`BlockQuantizationPipeline` drives `QuantizationPipeline` per block instead of per model, so only one block's activations and statistics are live. It keeps two streams: the float reference (original block on float inputs) and the student (adapted block on adapted inputs), passed between blocks as plain CPU tensor trees (`calibration/batches.py`). With `target_paths`, `QuantizationSchedule.trace` ranks targets by observed tensor dependencies, and each stage runs a fresh recipe restricted to that stage, so later stages collect through already adapted weights. The first stage pairs against the still-float block; a frozen float copy is made only when a later stage requests paired statistics. Teacher reconstruction (`GPTQEstimator(reconstruction="teacher")`) fits a ridge correction toward the teacher output before GPTQ and falls back to plain GPTQ if the fit fails. Usage and the trace contract are in `docs/block-calibration.md`.

### Block calibration of causal LMs (`causal_lm_blocks.py`)

`calibrate_causal_lm_blockwise(model, batches, config, input_source=...)` drives `BlockQuantizationPipeline.run_block` over a Hugging Face causal LM's decoder layers (`get_decoder().layers`), then over one more block holding the final norm and `lm_head` (and a tied input embedding, unused by its forward, so the adapter keeps the tie). That block's forward returns an empty slice of the logits: nothing reads them, and both streams of full-vocabulary logits would otherwise sit on the host (about 40 GB at 128 x 512 tokens and a 150k vocabulary). Modules the adapter replaces in the head block (mixed-precision splits) are installed on the model, and the splits each block records in its own manifest move to the model's, under full paths, so a saved checkpoint replays them; this holds for resumed blocks too. One float forward captures the first layer's hidden states and every layer's own keyword arguments (masks, rotary embeddings), so hybrid and sliding-window models get each layer's mask; tensors shared between layers are stored once, on the host. A model spread over several devices (an `hf_device_map` dispatch, or parameters on more than one device) is refused, since every block and finally the whole model move to one device. Targets (from `declare_targets()`; an estimator that cannot declare them is refused) outside the decoder layers and the head block (a tied input embedding belongs to the head block, so with `"sequential"` the layers see it float: quantizing it first would quantize the head's shared weight before the head's calibration) are quantized in one whole-model pass restricted to them (`QuantizationPipeline.run(target_paths=...)`), not checkpointed: before the capture with `"sequential"`, so the layers see them quantized, and after it with `"float"`, so they see the float model as in a whole-model run. `input_source="float"` reproduces a whole-model run's statistics; `"sequential"` calibrates on quantized upstream activations. `run_quantization_pipeline(..., block_calibration=...)` and `_make_entry(config, block_calibration=...)` select it. The option is not part of the recipe JSON, so cache fingerprints are unchanged; the manifest records it as `block_calibration` next to the recipe JSON, and `save_quantized_model` / `load_quantized_model` keep it. The block pipeline builds its recipe from `recipe_config_json`, so `run_quantization_pipeline` refuses a `recipe` whose settings differ from it (public attributes of the built estimator, adapter, calibration and preprocessing, through functions' code and closures, partials and bound methods' instances; kernel lowering aside). In that mode `cache_dir` holds block checkpoints (`<cache_dir>/blocks`, the `BlockQuantizationPipeline` checkpoints: adapted weights plus both output streams per block, about `2 * n_samples * seq_len * D` activations each) instead of per-estimator stages; a block resumes only when the recipe, mode, its float weights and its inputs all match, so a sequential block is recomputed whenever anything upstream changed. The span collector is not used. Block calibration runs parts of the recipe (blocks, the outside pass) and never applies its kernel lowering; the caller lowers once afterwards, as `run_quantization_pipeline` does. Running a whole recipe with `QuantizationPipeline.run` applies its lowering.

### Trellis quantization (`trellis_estimator.py`, `methods/trellis*.py`, `methods/block_ldlq.py`)

Bitshift-trellis coding in the style of QTIP / EXL3, in layers that can be reused separately:

- `kernels/trellis_viterbi.cu` + `kernels/trellis.py`: CUDA Viterbi encoder, decoder and unpacker, JIT-built on first use (needs `nvcc` and `ninja` on PATH). The encoder started from exllamav3's (MIT, Copyright (c) 2025 Turboderp; the notice is in the `.cu` header) and picks the same states for the rates both support.
- `methods/trellis_code.py`: `TrellisCode(bits, codebook, tile_len, lut)` codes `[n, tile_len]` value sequences at any rate from 1 to 8 bits in 1/16 steps; codebooks `mul1`, `mcg`, `3inst` (hash, no table) or `lut` (a 65536-entry fp16 table). Encoding needs CUDA; decode, pack and unpack also run in plain torch.
- `methods/block_ldlq.py`: LDLQ error feedback over blocks of columns with a pluggable block quantizer (`quantize_block(z, j)`), plus block act-order. Any quantizer that codes several columns jointly can use it.
- `methods/trellis.py`: the weight quantizer (random signs + 128-block Hadamard on both sides, fp16 row/column scales, global scale search, block LDLQ over 16x16 tiles) returning a `TrellisWeight`.
- `transforms/trellis.py`: `QTensorTrellis` (packed tiles + `TrellisScheme`), saved packed by `save_quantized_model` through its serde. `assign_as_parameter = True` makes the adapter assign it as the weight as is.
- `trellis_estimator.py`: estimator type `trellis`; per-module or per-type bits via `bits_per_group` (full name first, then leaf name), `store="packed"` (default) or `"dense"`. Sign seeds are `seed` plus the module's position among all linears of the model, so a staged `BlockQuantizationPipeline` run seeds every linear as a whole-model run does.

A packed weight is dequantized on every forward; for evaluation, dequantize the weights once (or use `store="dense"`).

### Pipeline wiring: `_cache` and `_spans`

`QuantizationParamEstimator` declares `_cache: CacheManager` and `_spans: SpanCollector` as class attributes with no-op defaults. The pipeline calls `set_cache()` and `set_span_collector()` before `prepare()`. Subclasses use `self._cache` / `self._spans` directly — no `getattr` fallback needed.

### Compression registry (`neuropress/t2n/registry.py`)

Follows the `torch_to_nnef` registry pattern: a dict (`NEUROPRESS_COMPRESSION`) maps string keys to callables with signature `fn(model, tokenizer=..., **kwargs) -> model`. Pass it to `LLMExporter.prepare`:

```python
exporter.prepare(
    compression_method="open_text_c4_gptq_q4_0",
    compression_registry="neuropress.t2n.registry.NEUROPRESS_COMPRESSION",
)
```

Every entry is built by `_make_entry(config)`, which binds a `QuantizationRecipeConfig` into a `functools.partial` of `run_quantization_pipeline` and stashes the config JSON on the partial as `__config__` so the manifest can record it. Calling an entry:

1. Reads `QuantizationRecipe.calibration` and loads batches from `calibration.DATASET_REGISTRY`
2. Transfers batches to the model device one at a time (no OOM on large calibration sets)
3. Delegates to `QuantizationPipeline`, which runs the six steps above
4. Applies kernel lowering once when the recipe configures one (`QuantizationPipeline.run` does it on the whole-model path, the entry itself after block calibration), except when `LLMExporter` calls it for an NNEF export (it passes `export_dirpath`), since tracing needs the QTensor modules

`save_quantized_model` unlowers a lowered model before writing, restoring fused RMSNorms to the modules they replaced (the fused weight may hold Gemma's folded `+1`), and lowers it again afterwards. `load_quantized_model` restores quantized tensors on the device of the parameters they replace, so a checkpoint loads into a CUDA model and lowers there.

Recipes can also be built from JSON via `QuantizationRecipeConfig.from_json(...).build()`, which resolves component types from the registries at deserialization time.

Layers whose `in_features` is not a multiple of the group size are skipped (Q4_0 group size = 32). Config `build()` rejects a `weight_transform` that cannot store the configured `nbits`, `symmetric` or `group_size`, before any estimation runs. GPTQ on a transformers `Conv1D` supports no named storage format: its scale groups run along the first weight dimension, which flat per-block storage cannot express without recalibrating.

Registry entries: `"open_text_c4_gptq_q4_0"` (128 samples), `"open_text_c4_gptq_q4_0_32samples"`, `"open_text_c4_gptq_q4_0_256samples"`, `"open_text_c4_gptq_q4_0_moe"` (also quantizes fused MoE experts, see below).

### Fused MoE experts (`moe_experts.py`, `calibration/probe/moe_experts.py`)

Transformers stores the routed experts of most MoE models as one module with 3-D `gate_up_proj [E, 2I, D]` and `down_proj [E, D, I]` parameters, which `GPTQEstimator` (`nn.Linear` only) does not see. Estimator type `gptq_moe_experts` covers them:

- `MoEExpertHessianCollector` (stat type `moe_expert_hessian`) is an `InvocationCollector`: the registry hands it the whole invocation `(hidden_states, top_k_index, top_k_weights)` rather than the first input. Per expert it accumulates the Hessian of the tokens routed to it and of `act(gate) * up`, recomputed from the float `gate_up_proj`. Tokens are grouped by expert with one host sync per call, every routed token counts once whatever its routing weight, and a layer's Hessians (both stacks together) move to the host when they would take a large share of free GPU memory (the `HessianAccumulator` rule); products then cross in batches of experts. `result()` scales the stacks in place and closes the collector.
- `GPTQMoEExpertsEstimator` runs GPTQ per expert (Q4_0, blocks of 32 along the reduction dimension; an expert no token reached is rounded to nearest, and one that saw fewer than `min_expert_tokens`, default 64, has its Hessian shrunk toward the identity) and plans one `QTensorTractScaleOnly` per fused parameter under `"<experts path>/gate_up_proj"` and `"<experts path>/down_proj"`. It reads a layer's Hessians where collection left them (no registry relocation) and moves one expert's copy at a time to the weights' device. Experts whose f16 scales overflow stay float, both parameters, like an `nn.Linear` would. `WeightAssignmentAdapter` ignores these keys, and a staged run (`target_paths`) counts them under the experts path. `neuropress-eval size` counts them as Q4_0 (4.5 bits per weight).
- `MoEExpertsQ40Adapter` (adapter type `moe_experts_q4_0`) writes the dequantized values into the float parameters and marks the parent MoE block for the torch_to_nnef `tract_moe_ffn` extractor: `_t2n_quantize_moe_experts_q40`, `_t2n_moe_expert_layout = "linear"` and a quantizer (`expert_q40_from_gptq_codes`) that returns the GPTQ codes instead of recalibrating with min-max. Only the f16 block scales are kept (on the experts module): the quantizer reads the codes back from each extracted `w1`/`w3`/`w2`, which must lie within a tenth of a step of the grid of one of them, and fails the export when none match. Experts at the top level mark the root module itself. A parent class torch_to_nnef has no MoE adapter for is left unmarked, with a warning. `save_quantized_model` stores the scales under a `moe_experts_q40` manifest transform, and `load_quantized_model` restores the markers.

The layout follows from the grouping: a Q4_0 block covers 32 consecutive values of the stored last dimension, and only the `linear` layout (`w1`/`w3 [E, I, D]`, `w2 [E, D, I]`) keeps each expert's input dimension last. `"open_text_c4_gptq_q4_0_moe"` chains this estimator with the Q4_0 linear GPTQ (weight transform `q4_0_packed_no_router`, which keeps routers and shared-expert gates float: leaves named `gate`, `router` or `shared_expert_gate`, and anything under a `gate` or `router` module), so an export through the compression registry needs no MoE-specific flag. A whole-model run holds `E * (D^2 + I^2)` float32 Hessian entries for every MoE layer at once (about 2.4 GB per layer for E=128, D=2048, I=768, so about 117 GB for Qwen3-30B-A3B), on the host once the GPU runs short. The registry entry therefore runs with `block_calibration="sequential"` (see below), which holds one layer's Hessians at a time, each layer calibrated on the outputs of the already quantized layers before it.

### LLM calibration data

`neuropress.calibration.data.llm.load_c4_calibration(tokenizer, ...)` loads C4 samples as `{"input_ids": Tensor[batch_size, seq_len]}` dicts. Requires the `llm` optional dependency (`datasets`).

### Integration test pattern

The integration test (`tests/integration/test_gptq_nnef_export.py`) demonstrates the end-to-end workflow: HuggingFace model load → C4 calibration → layer-by-layer GPTQ → `torch_to_nnef` NNEF export. It is gated behind `RUN_INTEGRATION=1` and also requires `transformers` and `datasets`.
