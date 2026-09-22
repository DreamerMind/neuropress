# neuropress

A graph-aware neural network quantization framework targeting NNEF export via
[`torch_to_nnef`](https://github.com/gomisc/torch-to-nnef).

## Installation

Requires Python 3.13 or newer.

```bash
pip install neuropress
pip install "neuropress[eval]"   # + lm-evaluation-harness
```

Extras: `llm` (HuggingFace datasets for C4 calibration), `eval`
(lm-evaluation-harness), `cloud` (vast.ai GPU runner), `audio`, `gemlite`, `vptq`.

### From source

Managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # core dependencies
uv sync --extra llm          # + HuggingFace datasets (C4 calibration)
uv sync --extra eval         # + lm-evaluation-harness
uv sync --extra cloud        # + vastai-sdk (GPU cloud runner)
uv sync --extra dev          # + pytest, ruff, ty
```

## Quick start

```bash
# Unit tests
uv run pytest

# Integration test — downloads ~500 MB model, runs GPTQ + NNEF export
RUN_INTEGRATION=1 uv run pytest tests/integration/ -v

# Estimate quantized model size — no model weights downloaded, no calibration
uv run neuropress-eval size google/gemma-3-270m \
    --compression open_text_c4_gptq_q4_0_256samples

uv run neuropress-eval size meta-llama/Llama-3.2-1B \
    --compression open_text_c4_gptq_q4_0_linear_hqq_q41_emb_256samples \
    --dtype bfloat16

# Evaluate a model (default device: CPU — no GPU required)
uv run neuropress-eval eval google/gemma-3-270m \
    --tasks hellaswag --limit 100

# Same on GPU
uv run neuropress-eval eval google/gemma-3-270m \
    --tasks hellaswag --limit 100 --device cuda

# Compare float vs quantized (GPTQ calibration + lm-eval)
uv run neuropress-eval compare google/gemma-3-270m \
    --tasks hellaswag arc_easy \
    --compression open_text_c4_gptq_q4_0 \
    --output report.json          # CPU (slow but works)

uv run neuropress-eval compare google/gemma-3-270m \
    --tasks hellaswag arc_easy \
    --compression open_text_c4_gptq_q4_0 \
    --device cuda --output report.json   # GPU (recommended)

# List available compression methods
uv run neuropress-eval list
```

### `neuropress-eval size` — static size estimation

Estimates the quantized model size **without downloading weights, running
calibration, or applying GPTQ**.  Only the model config JSON (a few KB) is
fetched; the model is instantiated on PyTorch's *meta* device (zero memory).

```bash
# Basic usage
neuropress-eval size google/gemma-3-270m \
    --compression open_text_c4_gptq_q4_0_256samples

# Specify baseline dtype (default: inferred from model config)
neuropress-eval size meta-llama/Llama-3.2-1B \
    --compression open_text_c4_gptq_q4_0_linear_hqq_q41_emb_256samples \
    --dtype bfloat16
```

The output is a colour-coded table (requires a colour terminal):

- **green** — layers with genuine storage reduction
- **yellow** — targeted layers stored as dequantized float (no size reduction)
- **dim** — untargeted layers and non-weight parameters (bias, LayerNorm)

The **bpw** column is effective bits per weight *including* scale/zero-point
overhead (e.g. Q4_0 packed at group_size=32 → 4 + 16/32 = **4.50 bpw**).

### Device selection

`--device` controls where the model is loaded and where calibration runs.
The default is **`cpu`** — no GPU is assumed or required.

| Flag | When to use |
|------|-------------|
| *(omit)* or `--device cpu` | CPU-only machines, quick tests, debugging |
| `--device cuda` | NVIDIA GPU (recommended for 256-sample GPTQ) |
| `--device cuda:1` | Specific GPU index |

> **CPU timing note (Gemma 3 270M, 256 calibration samples):**
> Hessian collection + GPTQ solve takes roughly **10–60 minutes** on CPU,
> depending on core count and BLAS quality (64-core server: ~10–15 min;
> 4-core laptop: ~45–60 min).  On an RTX 3090/4090 the same job finishes
> in **2–5 minutes**.

## Architecture

The full pipeline runs in six steps (see `neuropress/pipeline.py`):

| Step | Class | Role |
|------|-------|------|
| 0 | `PreprocessingStep` | Optional model transforms (e.g. SmoothQuant) run before tracing |
| 1 | `RuntimeModuleGraphProvider` | Hook-based forward-pass trace → `QGraph` |
| 2 | `WeightTarget` / `ActivationTarget` | Identify what to quantize |
| 3 | `StatisticsRegistry` | Collect calibration statistics (min/max, Hessian) |
| 4 | `QuantizationRecipe` | Ties estimator + adapter + allowed schemes |
| 5 | `QuantizationPlan` | Maps target keys → `ZpScaleScheme` / `QSchemeGroup` |
| 6 | `ModuleQuantizationAdapter` | Applies the plan to the model |

### Key abstractions

| Class | Role |
|-------|------|
| `QSchemeDescriptor` | Static backend constraint (nbits, symmetric, granularity) |
| `ZpScaleScheme` | Concrete affine scheme (zero_point, scale, nbits) |
| `QSchemeGroup` / `PanelSpec` | Mixed-precision grouping of weight regions |
| `GPTQCalibration` | Full GPTQ implementation (Frantar et al. ICLR 2023) |
| `GPTQMixedPrecisionCalibration` | GPTQ with per-column-group bit-width allocation |
| `MixedPrecisionLinear` | Drop-in replacement for `nn.Linear` |
| `GPTQEstimator` | Dataclass estimator; `prepare(model)` discovers linear layers |
| `WeightAssignmentAdapter` | Applies `plan.weight_assignments` via `ModTensorUpdater` |

### Compression registry

```python
from neuropress.t2n import NEUROPRESS_COMPRESSION

# Use with torch_to_nnef LLMExporter
exporter.prepare(
    compression_method="open_text_c4_gptq_q4_0",
    compression_registry="neuropress.t2n.NEUROPRESS_COMPRESSION",
)
```

See all keys with `neuropress-eval list`.  Built-in entries:

| Key | Method |
|-----|--------|
| `open_text_c4_gptq_q4_0_256samples` | GPTQ Q4_0 linear (C4, 256 samples) |
| `open_text_c4_gptq_q4_0_linear_minmax_q4_0_emb_256samples` | GPTQ Q4_0 linear + min-max Q4_0 embedding |
| `open_text_c4_gptq_q4_0_linear_hqq_q41_emb_256samples` | GPTQ Q4_0 linear + HQQ asymmetric Q4 embedding |
| `open_text_c4_gptq_panel_q4q2_linear_256samples` | GPTQ mixed Q4/Q2 linear (3.0 avg bpw) |
| `speech_librispeech_gptq_q4_0_256samples` | GPTQ Q4_0 (LibriSpeech, 256 samples) |

### Ternary PTQ for vLLM

NeuroPress includes a Bonsai-style path that combines blockwise signed
Hadamard preprocessing, strict ternary GPTQ, group-128 FP16 scales, and export
to the [`prism_ternary`](https://github.com/fraserprice/bonsai-vllm) vLLM
checkpoint layout.

```python
from neuropress import QuantizationPipeline, QuantizationRecipeConfig
from neuropress.formats import export_prism_ternary_checkpoint

recipe = QuantizationRecipeConfig.model_validate(
    {
        "calibration": {"dataset": "c4", "seq_len": 2048, "n_samples": 128},
        "preprocessing": [
            {"type": "ternary_hadamard", "block_size": 1024, "seed": 1234}
        ],
        "estimator": {
            "type": "ternary_gptq",
            "group_size": 128,
            "threshold_ratio": 0.4,
        },
        "kernel_lowering": None,
    }
).build()

quantized = QuantizationPipeline(recipe).run(model, calibration_batches)
export_prism_ternary_checkpoint(
    quantized,
    "./model-ternary-vllm",
    base_config="./base-hf-checkpoint",
)
```

The exported checkpoint stores codes `0/1/2` as 16 two-bit lanes per int32,
uses one FP16 scale for every 128 columns, and embeds the shared sign vectors
in `config.json`. Code `3` is reserved and never emitted. Install the vLLM
plugin before serving the result. The exporter deliberately refuses ordinary
2-bit affine weights; its input must already be strict ternary after GPTQ.
The `0.4` threshold is an empirical Bonsai-format match, not a published Prism
hyperparameter; use `0.5` for conventional nearest-level ternary rounding and
sweep this value against perplexity for a new architecture.

For large models, the ternary recipe shares Hessians between sibling
projections with identical inputs and offloads each unique Hessian between
calibration updates. This bounds resident memory, but quantizing a 27B-class
model still requires substantial temporary disk space (roughly the sum of the
unique FP32 Hessian sizes). Point the system temporary directory at a volume
with enough free space. Export is transactional and packs weights in row
chunks, so a failed validation does not leave a partial checkpoint.

The asynchronous Hessian prefetch path uses the scoped residency API merged in
`torch-to-nnef` PR 354. Until that API appears in a release after 0.24.8,
`uv sync` from this repository uses the pinned upstream commit in
`tool.uv.sources`. Installations resolved only from released package metadata
remain correct but log a warning and reload Hessians synchronously.

---

## Cloud GPU runner (`neuropress-cloud`)

Run quantization and evaluation jobs on [vast.ai](https://cloud.vast.ai) spot
GPUs with **guaranteed instance cleanup** — the instance is always destroyed on
exit, even on `KeyboardInterrupt` or unhandled exceptions.

> **Billing note:** vast.ai charges **per second** from instance creation.
> A 10-minute job on an RTX 4090 at $0.35/hr costs ~$0.10 total.

### Prerequisites

```bash
# Install the cloud extra
uv sync --extra cloud

# Register your SSH public key with vast.ai (one-time)
vastai create ssh-key "$(cat ~/.ssh/id_rsa.pub)"

# Set credentials
export VAST_API_KEY=your_key_here
export HF_TOKEN=hf_xxxxxxxxxxxx   # required for gated models (e.g. Gemma)
```

Your vast.ai API key is at <https://cloud.vast.ai/account/>.
Your HuggingFace token is at <https://huggingface.co/settings/tokens>
(needs Read access; request model access at the model page first).

### Basic usage

```bash
# List available built-in jobs
neuropress-cloud --list-jobs

# Preview GPU offers without renting anything
neuropress-cloud --job neuropress-integration --dry-run

# Run integration tests (GPTQ calibration + NNEF export on Gemma-3-270M)
neuropress-cloud --job neuropress-integration

# Run lm-eval harness: compare float vs gptq_q4_0
uv run neuropress-cloud --job neuropress-eval-compare \
      --model google/gemma-3-270m \
      --compression open_text_c4_gptq_q4_0_256samples \
      --tasks hellaswag --limit 100
```

Artifacts are downloaded to `./vast_output/<job>/` (or `--output DIR/<job>/`).
On failure, `./vast_output/<job>/debug_job.log` contains the full remote log.

### What it does

**Fresh run:**

```
1. search_offers   — find cheapest GPU matching the query + host filters
2. create_instance — rent it (billing starts now)
3. poll SSH        — wait until the Docker image is pulled and SSH is ready
4. rsync →         — sync project to remote (delta, mirrors local exactly)
5. setup           — apt-get build-essential, install uv, uv sync
6. run             — job commands in a detached tmux session
7. rsync ←         — download artifacts
8. destroy         — billing stops  ← always runs, even on crash
```

**Reuse run** (`--instance-id`):

```
1. attach          — look up existing instance, probe SSH
2. rsync →         — sync only changed files (--delete mirrors local)
3. setup           — uv sync (fast: venv already exists)
4. run
5. rsync ←
6. destroy  (or keep if --keep)
```

### Reusing instances

Docker image pulls (~1–2 min) and `uv sync` on a cold venv (~3–5 min) are
billed idle time.  Use `--keep` to preserve the instance between runs:

```bash
# Run 1 — keep the instance alive after the job
neuropress-cloud --job neuropress-integration --keep
# → [keep] Instance 32250248 preserved.
# →   Reuse next time: --instance-id 32250248

# Run 2 — attach to the warm instance (skips image pull + venv setup)
neuropress-cloud --job neuropress-integration --instance-id 32250248 --keep

# Final run — attach and destroy when done
neuropress-cloud --job neuropress-integration --instance-id 32250248
```

If the previous run crashed and you fixed something locally, rsync will detect
the changes and re-upload only the modified files.  Deleted/renamed files are
also cleaned up on the remote (`--delete` is always passed to rsync uploads).

### Host quality filters

By default the runner filters for hosts with ≥ 500 Mbps download and
≥ 0.9 reliability.  These can be tuned or combined with privacy flags:

```bash
# Datacenter hosts only (colocation, not residential)
neuropress-cloud --job neuropress-integration --datacenter

# Identity-verified hosts (vast.ai KYC check)
neuropress-cloud --job neuropress-integration --verified

# Combine both for maximum trust
neuropress-cloud --job neuropress-integration --datacenter --verified

# Require faster download (shorter image-pull time)
neuropress-cloud --job neuropress-integration --min-inet-down 1000

# Check what's available before committing
neuropress-cloud --job neuropress-integration --datacenter --dry-run
```

> **Privacy note:** vast.ai is a marketplace — hosts are third parties with
> physical access to their hardware.  Docker provides software isolation only.
> For public models and open-source code this is fine.  For proprietary weights,
> prefer a dedicated cloud provider with confidential computing support.

### Reducing cold-start time

A fresh instance has a ~15 min overhead before the job starts:

| Phase | Default | With pre-baked image |
|-------|---------|---------------------|
| Docker image pull | ~8 min (`pytorch/pytorch` ≈ 5 GB) | ~1 min (`neuropress-base` ≈ 800 MB) |
| apt + uv install | ~5 min | ~5 min (still needed first time) |
| uv sync | ~5 min (downloads torch + deps) | ~10 sec (cache already in image) |
| **Total cold-start** | **~15–18 min** | **~3–4 min** |

**Build the pre-baked image once** (requires Docker Hub account):

```bash
docker build -f Dockerfile.vast -t <dockerhub-user>/neuropress-base:latest .
docker push <dockerhub-user>/neuropress-base:latest   # must be public
```

Then set `NEUROPRESS_DOCKER_IMAGE` before running any job:

```bash
export NEUROPRESS_DOCKER_IMAGE=<dockerhub-user>/neuropress-base:latest
neuropress-cloud --job neuropress-eval-compare ...
```

Rebuild the image whenever `pyproject.toml` dependencies change.

**Alternative: reuse instances** — `--keep` preserves the Docker image and warm
uv cache, reducing subsequent runs to ~1–2 min overhead regardless of image.

### GPU cost estimate

Billed per second.  Estimates for `neuropress-integration` (Gemma-3-270M):

| | First run | Reuse (`--instance-id`) |
|---|---|---|
| Setup overhead | ~6 min (image pull + uv sync) | ~1 min (uv sync only) |
| RTX 4090 @ $0.35/hr | ~$0.15 total | ~$0.10 total |
| RTX 3090 @ $0.20/hr | ~$0.10 total | ~$0.07 total |

### `--cloud own` — run on any machine (no vast.ai account needed)

Use `--cloud own` to run any job on a machine you already have SSH access to —
a home PC, a university cluster, or any remote server.  No vast.ai API key is
required.  The machine is **never destroyed**.

```bash
# Local PC (GTX 1080, Ubuntu)
neuropress-cloud --cloud own \
    --ssh-host 192.168.1.42 --ssh-user myuser \
    --job neuropress-eval-compare \
    --model google/gemma-3-270m \
    --compression open_text_c4_gptq_q4_0_256samples \
    --tasks hellaswag --limit 100 \
    --device cuda \
    --hf-token $HF_TOKEN

# Remote server with non-standard SSH port
neuropress-cloud --cloud own \
    --ssh-host my-server.example.com --ssh-port 2222 --ssh-user ubuntu \
    --job neuropress-integration
```

The setup commands (`apt-get`, `uv sync`, …) are the same as for vast.ai and
are idempotent — safe to re-run on every invocation.  After the first run the
uv cache is warm and subsequent runs skip all downloads.

> **Note:** `--keep`, `--instance-id`, `--dry-run`, and GPU host-filter flags
> (`--datacenter`, `--verified`, …) are ignored for `--cloud own`.

### Adding a custom job

```python
from neuropress.cloud.vast_runner import VastJob, GpuSpec, _uv_setup, _JOB_REGISTRY
from pathlib import Path

def job_my_experiment(project_root: Path, output_dir: Path) -> VastJob:
    remote = "/workspace/neuropress"
    return VastJob(
        name="my-experiment",
        gpu=GpuSpec(
            query="gpu_name=RTX_4090 num_gpus=1 rented=False rentable=True dph<0.5",
        ),
        uploads=[project_root],
        remote_workdir=remote,
        setup=_uv_setup(remote),
        run=[
            "~/.local/bin/uv run python my_script.py 2>&1 | tee /tmp/out.log"
            " ; exit ${PIPESTATUS[0]}",
        ],
        artifacts=["/tmp/out.log"],
        output_dir=output_dir / "my-experiment",
        required_env_vars=["HF_TOKEN"],   # omit if no gated models
    )

_JOB_REGISTRY["my-experiment"] = job_my_experiment
```

Then run with `neuropress-cloud --job my-experiment`.

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--cloud BACKEND` | `vastai` | Backend: `vastai` (rent GPU) or `own` (direct SSH to your machine) |
| `--api-key KEY` | `$VAST_API_KEY` | vast.ai API key (required for `--cloud vastai`) |
| `--ssh-key PATH` | `~/.ssh/id_rsa` | SSH private key |
| `--job NAME` | — | Job to run (required unless `--list-jobs`) |
| `--dry-run` | off | Preview matching GPU offers, rent nothing (`vastai` only) |
| `--list-jobs` | off | Print available job names and exit |
| `--output DIR` | `./vast_output` | Local directory for downloaded artifacts |
| `--project-root DIR` | repo root | Local neuropress project path |
| **Instance reuse** (`vastai` only) | | |
| `--keep` | off | Do not destroy instance after job; print ID for `--instance-id` |
| `--instance-id ID` | — | Attach to an existing instance (skips rent + image pull) |
| **Gated models** | | |
| `--hf-token TOKEN` | `$HF_TOKEN` | HuggingFace token for gated models (e.g. Gemma) |
| **Host filters** (`vastai` only) | | |
| `--datacenter` | off | Restrict to datacenter/colocation hosts |
| `--verified` | off | Restrict to identity-verified hosts |
| `--min-inet-down MBPS` | `500` | Minimum host download bandwidth |
| `--min-reliability SCORE` | `0.9` | Minimum vast.ai reliability score (0–1) |
| **Job parameters** | | |
| `--model HF_ID` | — | HuggingFace model ID (eval jobs) |
| `--compression METHOD` | — | Compression method key (eval jobs) |
| `--tasks TASK ...` | — | lm-eval task names (eval jobs) |
| `--limit N` | all | Max examples per task |
| `--num-fewshot K` | task default | Few-shot count |
| `--device DEVICE` | `cuda` | PyTorch device for eval jobs (e.g. `cpu`, `cuda`, `cuda:1`) |
| **Direct SSH** (`--cloud own`) | | |
| `--ssh-host HOST` | — | Hostname or IP of the target machine (required for `--cloud own`) |
| `--ssh-port PORT` | `22` | SSH port on the target machine |
| `--ssh-user USER` | `root` | SSH login user |
