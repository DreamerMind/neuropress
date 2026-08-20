"""LLM calibration data loaders.

Three loaders at distinct abstraction levels:

``load_c4_calibration``
    Generic web text (C4).  The quantization-paper standard — blind to any
    downstream eval task.

``load_fineweb_calibration``
    Generic web text (FineWeb).  Higher quality and more diverse than C4
    (books, academic, code represented in the crawl mix).  Same interface;
    drop-in replacement for C4 when a richer generic corpus is preferred.
    Still fully blind to eval tasks.

``load_eval_targeted_calibration``
    Mixes C4 with eval-task *train* splits formatted as complete sequences
    (prompt + gold answer).  Explicitly eval-distribution-aware — use only
    when you intend to calibrate for a specific task set.  **Not standard
    calibration practice**; results are not directly comparable to papers
    that use generic-text calibration.

Install the optional dependency::

    pip install neuropress[llm]   # adds `datasets`

Each loader yields ``{"input_ids": Tensor[batch_size, seq_len]}`` dicts,
ready to be unpacked by ``QuantizationPipeline.run`` or
``StatisticsRegistry.collect``.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Task format definitions for mixed calibration
# ---------------------------------------------------------------------------

# Each entry: (dataset_id, dataset_config_or_None, split, format_fn)
# format_fn(example) -> str: complete text including the gold answer
_TASK_CONFIGS: dict[
    str, tuple[str, str | None, str, Callable[[dict[str, Any]], str]]
] = {
    "boolq": (
        "aps/super_glue",
        "boolq",
        "train",
        lambda ex: (
            f"{ex['passage']}\nQuestion: {ex['question']}?\n"
            f"Answer: {'yes' if ex['label'] == 1 else 'no'}"
        ),
    ),
    "arc_easy": (
        "allenai/ai2_arc",
        "ARC-Easy",
        "train",
        lambda ex: (
            "Question: "
            + ex["question"]
            + "\nAnswer: "
            + next(
                t
                for label, t in zip(
                    ex["choices"]["label"], ex["choices"]["text"], strict=True
                )
                if label == ex["answerKey"]
            )
        ),
    ),
    "arc_challenge": (
        "allenai/ai2_arc",
        "ARC-Challenge",
        "train",
        lambda ex: (
            "Question: "
            + ex["question"]
            + "\nAnswer: "
            + next(
                t
                for label, t in zip(
                    ex["choices"]["label"], ex["choices"]["text"], strict=True
                )
                if label == ex["answerKey"]
            )
        ),
    ),
    "piqa": (
        "baber/piqa",
        None,
        "train",
        lambda ex: (
            f"Question: {ex['goal']}\n"
            f"Answer: {ex['sol1'] if ex['label'] == 0 else ex['sol2']}"
        ),
    ),
    "openbookqa": (
        "allenai/openbookqa",
        "main",
        "train",
        lambda ex: (
            ex["question_stem"]
            + " "
            + next(
                t
                for label, t in zip(
                    ex["choices"]["label"], ex["choices"]["text"], strict=True
                )
                if label == ex["answerKey"].strip()
            )
        ),
    ),
}


def load_c4_calibration(
    tokenizer: Any,
    *,
    seq_len: int = 512,
    n_samples: int = 128,
    batch_size: int = 1,
    seed: int = 42,
    split: str = "train",
    dataset_id: str = "allenai/c4",
    dataset_config: str = "en",
    streaming: bool = True,
    shuffle_buffer_size: int = 1_000,
    _load_dataset: Any | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Load calibration batches from the C4 dataset.

    Texts are tokenized, concatenated into a running buffer, and sliced into
    non-overlapping chunks of exactly *seq_len* tokens.  *n_samples* such
    chunks are collected, shuffled, then yielded as batches lazily — only one
    ``{"input_ids": Tensor}`` dict is in memory at a time.

    Args:
        tokenizer: Any HuggingFace-compatible tokenizer with a ``__call__``
            method that returns an object with an ``input_ids`` attribute.
        seq_len: Token sequence length per sample — 512 or 1024 are typical.
        n_samples: Total number of *seq_len*-length samples to collect.
            Must be divisible by *batch_size* for all batches to be full;
            the last incomplete batch is dropped.
        batch_size: Samples per yielded dict.
        seed: Random seed used to shuffle the stream and the final sample list.
        split: Dataset split — ``"train"`` or ``"validation"``.
        dataset_id: HuggingFace dataset hub identifier.
        dataset_config: Dataset configuration / language subset.
        streaming: If ``True``, stream the dataset without a full download.
            The stream is shuffled via a buffer when the dataset object
            supports it.
        shuffle_buffer_size: Size of the streaming shuffle buffer.  Larger
            values give more diverse sampling but force more shards to be
            fetched (each C4 shard is a large non-seekable gzip file, so
            filling the buffer means downloading whole shards) before the
            first sample is even yielded.  On a slow or throttled host this
            can push total download time past the ~1hr TTL on HF's presigned
            shard URLs, causing a 403.  Kept well below the datasets-library
            default (10_000) since *n_samples* here is typically small.
        _load_dataset: Injection point for testing.  When ``None`` the real
            ``datasets.load_dataset`` is imported lazily.

    Yields:
        ``{"input_ids": Tensor[batch_size, seq_len]}`` dicts.

    Raises:
        ImportError: If ``datasets`` is not installed and *_load_dataset* is
            not provided.
        RuntimeError: If not enough text is available to fill *n_samples*.
    """
    if _load_dataset is None:
        try:
            from datasets import load_dataset as _hf_load  # type: ignore[import]

            _load_dataset = _hf_load
        except ImportError as exc:
            raise ImportError(
                "HuggingFace `datasets` is required for LLM calibration data.\n"
                "Install it with:  pip install neuropress[llm]"
            ) from exc

    dataset = _load_dataset(
        dataset_id,
        dataset_config,
        split=split,
        streaming=streaming,
    )

    # Shuffle at the stream level for diversity; fall back gracefully when the
    # iterable does not support it (e.g. plain lists in tests).
    if streaming and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    rng = random.Random(seed)
    token_buffer: list[int] = []
    samples: list[torch.Tensor] = []

    for example in dataset:
        if len(samples) >= n_samples:
            break
        text: str = example.get("text", "")
        if not text:
            continue
        ids: list[int] = tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        token_buffer.extend(ids)
        # Drain complete seq_len chunks from the buffer.
        while len(token_buffer) >= seq_len and len(samples) < n_samples:
            samples.append(torch.tensor(token_buffer[:seq_len], dtype=torch.long))
            token_buffer = token_buffer[seq_len:]

    if len(samples) < n_samples:
        raise RuntimeError(
            f"C4 calibration: only {len(samples)}/{n_samples} samples could be "
            f"collected at seq_len={seq_len}.  Try a larger split or reduce n_samples."
        )

    rng.shuffle(samples)

    # Yield full batches lazily — only one dict is allocated at a time.
    for i in range(0, len(samples) - batch_size + 1, batch_size):
        yield {"input_ids": torch.stack(samples[i : i + batch_size])}


def load_fineweb_calibration(
    tokenizer: Any,
    *,
    seq_len: int = 512,
    n_samples: int = 128,
    batch_size: int = 1,
    seed: int = 42,
    dataset_config: str = "sample-10BT",
    _load_dataset: Any | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Load calibration batches from FineWeb.

    FineWeb (HuggingFace/FineWeb) is a higher-quality, more diverse web
    corpus than C4 — the crawl mix includes books, academic text, and
    technical content underrepresented in C4.  Same interface as
    :func:`load_c4_calibration`; blind to any downstream eval task.

    Args:
        tokenizer: Any HuggingFace-compatible tokenizer.
        seq_len: Token sequence length per sample.
        n_samples: Total number of seq_len-length samples to collect.
        batch_size: Samples per yielded dict.
        seed: Random seed.
        dataset_config: FineWeb sample size config.  ``"sample-10BT"`` is
            the smallest slice (10B tokens); ``"sample-100BT"`` for more
            diversity at the cost of a larger download.
        _load_dataset: Injection point for testing.

    Yields:
        ``{"input_ids": Tensor[batch_size, seq_len]}`` dicts.
    """
    yield from load_c4_calibration(
        tokenizer,
        seq_len=seq_len,
        n_samples=n_samples,
        batch_size=batch_size,
        seed=seed,
        dataset_id="HuggingFace/FineWeb",
        dataset_config=dataset_config,
        _load_dataset=_load_dataset,
    )


def _collect_task_samples(
    task_name: str,
    tokenizer: Any,
    n_samples: int,
    seq_len: int,
    seed: int,
    load_dataset: Any,
) -> list[torch.Tensor]:
    """Tokenize a task's train split and slice into seq_len chunks.

    Examples are formatted as complete sequences (prompt + gold answer) so the
    Hessian captures activations over answer tokens, not just the prompt.
    Cycles through the dataset if needed to fill *n_samples* chunks.
    """
    dataset_id, dataset_config, split, format_fn = _TASK_CONFIGS[task_name]
    kwargs: dict[str, Any] = {"split": split}
    if dataset_config is not None:
        dataset = load_dataset(dataset_id, dataset_config, **kwargs)
    else:
        dataset = load_dataset(dataset_id, **kwargs)

    rng = random.Random(seed)
    examples = list(dataset)
    rng.shuffle(examples)

    token_buffer: list[int] = []
    samples: list[torch.Tensor] = []
    idx = 0

    while len(samples) < n_samples:
        ex = examples[idx % len(examples)]
        idx += 1
        try:
            text = format_fn(ex)
        except (StopIteration, KeyError):
            continue
        ids: list[int] = tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        token_buffer.extend(ids)
        while len(token_buffer) >= seq_len and len(samples) < n_samples:
            samples.append(torch.tensor(token_buffer[:seq_len], dtype=torch.long))
            token_buffer = token_buffer[seq_len:]

    return samples


def load_eval_targeted_calibration(
    tokenizer: Any,
    *,
    seq_len: int = 512,
    n_samples: int = 128,
    batch_size: int = 1,
    seed: int = 42,
    c4_fraction: float = 0.75,
    task_names: tuple[str, ...] = (
        "boolq",
        "arc_challenge",
        "arc_easy",
        "openbookqa",
        "piqa",
    ),
    _load_dataset: Any | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Load calibration batches mixing C4 with eval-task train splits.

    **Not standard calibration practice.** Explicitly eval-distribution-aware:
    25% of samples are drawn from the train splits of *task_names*, formatted
    as complete sequences (prompt + gold answer), so the Hessian sees the
    activation patterns of those specific tasks.  Results are not directly
    comparable to papers using generic-text calibration.

    Use this only when you intend to calibrate for a specific known task set
    and accept the benchmark-awareness tradeoff.  For general-purpose
    calibration prefer :func:`load_c4_calibration` or
    :func:`load_fineweb_calibration`.

    Args:
        tokenizer: Any HuggingFace-compatible tokenizer.
        seq_len: Token sequence length per sample.
        n_samples: Total number of seq_len-length samples to collect.
        batch_size: Samples per yielded dict.
        seed: Random seed for shuffling.
        c4_fraction: Fraction of samples drawn from C4.  The remainder is
            split equally across *task_names*.
        task_names: Tasks to include.  Must be keys in ``_TASK_CONFIGS``.
            Only train splits are used — no eval-split contamination.
        _load_dataset: Injection point for testing.

    Yields:
        ``{"input_ids": Tensor[batch_size, seq_len]}`` dicts.

    Raises:
        ImportError: If ``datasets`` is not installed.
        ValueError: If an unknown task name is provided.
    """
    unknown = [t for t in task_names if t not in _TASK_CONFIGS]
    if unknown:
        raise ValueError(
            f"Unknown task names: {unknown}. Available: {sorted(_TASK_CONFIGS)}"
        )

    if _load_dataset is None:
        try:
            from datasets import load_dataset as _hf_load  # type: ignore[import]

            _load_dataset = _hf_load
        except ImportError as exc:
            raise ImportError(
                "HuggingFace `datasets` is required for LLM calibration data.\n"
                "Install it with:  pip install neuropress[llm]"
            ) from exc

    n_c4 = round(n_samples * c4_fraction)
    n_task_total = n_samples - n_c4
    n_per_task = max(1, n_task_total // len(task_names)) if task_names else 0

    # C4 samples via existing loader (returns an iterator — drain it).
    c4_samples: list[torch.Tensor] = []
    for batch in load_c4_calibration(
        tokenizer,
        seq_len=seq_len,
        n_samples=n_c4,
        batch_size=1,
        seed=seed,
        _load_dataset=_load_dataset,
    ):
        c4_samples.append(batch["input_ids"].squeeze(0))

    # Task samples.
    task_samples: list[torch.Tensor] = []
    for task in task_names:
        task_samples.extend(
            _collect_task_samples(
                task, tokenizer, n_per_task, seq_len, seed, _load_dataset
            )
        )

    all_samples = c4_samples + task_samples
    random.Random(seed).shuffle(all_samples)
    all_samples = all_samples[:n_samples]

    for i in range(0, len(all_samples) - batch_size + 1, batch_size):
        yield {"input_ids": torch.stack(all_samples[i : i + batch_size])}
