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

1. **QGraph** – `RuntimeModuleGraphProvider` traces a single forward pass using PyTorch module hooks to build a `QGraph` (`nodes: Dict[str, QNode]`, `values: Dict[str, QValue]`). No symbolic tracing (FX/Dynamo) is used.

2. **Targets** – `WeightTarget` (`module_id/param_name`) and `ActivationTarget` (`value_id`) identify what to quantize.

3. **StatisticsRegistry** – `QuantizationParamEstimator.declare_statistics()` returns `StatisticRequest` objects. The registry deduplicates them (key = `target_key::stat_type`), installs hooks, runs a forward pass over calibration data, and stores results. Supported stat types include `"min_max"`, `"hessian"` and the paired `"linear_reconstruction"`. A `PairedCollector` also receives the matching output of a teacher model run on aligned `reference_data`; the pipeline builds a teacher only if `requires_pairing()` says some request needs one. With an `estimation_device`, `registry.to()` makes `get()` return each statistic on that device at access time, one at a time, while `inject()` hands statistics over as collected (GPTQ places each Hessian itself). The estimator is prepared again only if moving to the estimation device actually relocates the model. `collection_model` lets statistics come from an inference view of the model, and `target_paths` scopes the estimator through `select_targets()` and rejects assignments outside that set.

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
4. Applies kernel lowering when the recipe configures one, except when `LLMExporter` calls it for an NNEF export (it passes `export_dirpath`), since tracing needs the QTensor modules

`save_quantized_model` unlowers a lowered model before writing, restoring fused RMSNorms to the modules they replaced (the fused weight may hold Gemma's folded `+1`), and lowers it again afterwards. `load_quantized_model` restores quantized tensors on the device of the parameters they replace, so a checkpoint loads into a CUDA model and lowers there.

Recipes can also be built from JSON via `QuantizationRecipeConfig.from_json(...).build()`, which resolves component types from the registries at deserialization time.

Layers whose `in_features` is not a multiple of the group size are skipped (Q4_0 group size = 32). Config `build()` rejects a `weight_transform` that cannot store the configured `nbits`, `symmetric` or `group_size`, before any estimation runs. GPTQ on a transformers `Conv1D` supports no named storage format: its scale groups run along the first weight dimension, which flat per-block storage cannot express without recalibrating.

Registry entries: `"open_text_c4_gptq_q4_0"` (128 samples), `"open_text_c4_gptq_q4_0_32samples"`, `"open_text_c4_gptq_q4_0_256samples"`.

### LLM calibration data

`neuropress.calibration.data.llm.load_c4_calibration(tokenizer, ...)` loads C4 samples as `{"input_ids": Tensor[batch_size, seq_len]}` dicts. Requires the `llm` optional dependency (`datasets`).

### Integration test pattern

The integration test (`tests/integration/test_gptq_nnef_export.py`) demonstrates the end-to-end workflow: HuggingFace model load → C4 calibration → layer-by-layer GPTQ → `torch_to_nnef` NNEF export. It is gated behind `RUN_INTEGRATION=1` and also requires `transformers` and `datasets`.
