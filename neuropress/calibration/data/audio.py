"""Optional audio calibration data loader using HuggingFace LibriSpeech.

Install the optional dependency::

    pip install neuropress[audio]   # adds `datasets` and `soundfile`

Typical use::

    from transformers import AutoFeatureExtractor
    from neuropress.calibration.data import load_librispeech_calibration

    processor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
    batches = load_librispeech_calibration(processor, seq_len=480_000, n_samples=128)
    pipeline.run(model, batches)

Audio clips are loaded from the ``train.100`` split of LibriSpeech, truncated
to *seq_len* samples, **sorted ascending by clip length** (shorter clips first)
for stable padding within each batch, then processed by the caller-supplied
*processor*.  Only one batch dict is alive at a time.

Each yielded batch is a plain ``dict[str, Tensor]`` produced by *processor*
(e.g. ``{"input_values": Tensor[batch_size, seq_len]}`` for Wav2Vec2 or
``{"input_features": Tensor[batch_size, 80, frames]}`` for Whisper), ready
to be unpacked by ``QuantizationPipeline.run`` or
``StatisticsRegistry.collect``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch


def load_librispeech_calibration(
    processor: Any,
    *,
    seq_len: int = 480_000,
    n_samples: int = 128,
    batch_size: int = 8,
    seed: int = 42,
    split: str = "train.100",
    dataset_config: str = "clean",
    dataset_id: str = "openslr/librispeech_asr",
    sampling_rate: int = 16_000,
    streaming: bool = True,
    _load_dataset: Any | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Load calibration batches from the LibriSpeech dataset.

    Audio clips are streamed from HuggingFace, truncated to *seq_len* samples,
    collected until *n_samples* clips are available, then **sorted ascending by
    clip length** for stable within-batch padding.  The caller-supplied
    *processor* converts each batch of raw waveforms to model-ready tensors.

    Args:
        processor: Any HuggingFace-compatible feature extractor / processor
            whose ``__call__`` accepts a list of ``numpy`` arrays plus
            ``sampling_rate`` and ``return_tensors="pt"`` keyword arguments.
        seq_len: Maximum waveform length in samples.  Clips longer than this
            are truncated; shorter clips are padded by *processor*.  At 16 kHz,
            ``480_000`` corresponds to 30 s (Whisper's native window).
        n_samples: Total number of audio clips to collect.  The last incomplete
            batch is dropped, so the effective sample count is
            ``(n_samples // batch_size) * batch_size``.
        batch_size: Clips per yielded batch.
        seed: Unused (kept for API symmetry with :func:`.load_c4_calibration`);
            ordering is deterministic via duration sort.
        split: LibriSpeech dataset split — ``"train.100"``, ``"train.360"``,
            or ``"validation"``.
        dataset_config: Dataset configuration — ``"clean"`` or ``"other"``.
        dataset_id: HuggingFace dataset hub identifier.
        sampling_rate: Expected audio sampling rate in Hz.  Clips whose native
            rate differs are not resampled here; ensure *processor* handles
            resampling if needed.
        streaming: If ``True``, stream without a full download.
        _load_dataset: Injection point for testing.  When ``None`` the real
            ``datasets.load_dataset`` is imported lazily.

    Yields:
        ``dict[str, Tensor]`` produced by *processor* for each batch of clips,
        e.g. ``{"input_values": Tensor[batch_size, seq_len]}`` (Wav2Vec2) or
        ``{"input_features": Tensor[batch_size, 80, frames]}`` (Whisper).

    Raises:
        ImportError: If ``datasets`` (or ``soundfile``) is not installed and
            *_load_dataset* is not provided.
        RuntimeError: If not enough audio clips are available to fill
            *n_samples*.
    """
    if _load_dataset is None:
        try:
            from datasets import load_dataset as _hf_load  # type: ignore[import]

            _load_dataset = _hf_load
        except ImportError as exc:
            raise ImportError(
                "HuggingFace `datasets` and `soundfile` are required for audio "
                "calibration data.\nInstall them with:  pip install neuropress[audio]"
            ) from exc

    dataset = _load_dataset(
        dataset_id,
        dataset_config,
        split=split,
        streaming=streaming,
        trust_remote_code=False,
    )

    # Collect up to n_samples raw waveform arrays, truncated to seq_len.
    clips: list[np.ndarray] = []
    for example in dataset:
        if len(clips) >= n_samples:
            break
        audio = example.get("audio")
        if audio is None:
            continue
        arr: np.ndarray = np.asarray(audio["array"], dtype=np.float32)
        if arr.size == 0:
            continue
        # Truncate to seq_len; shorter clips are padded later by the processor.
        clips.append(arr[:seq_len])

    if len(clips) < n_samples:
        raise RuntimeError(
            f"LibriSpeech calibration: only {len(clips)}/{n_samples} clips could "
            f"be collected from split={split!r}, config={dataset_config!r}.  "
            f"Try a larger split or reduce n_samples."
        )

    # Sort ascending by clip length — shorter first — for stable within-batch
    # padding (mirrors the reference nemo_asr calibration pattern).
    clips.sort(key=lambda a: a.size)

    # Yield full batches lazily — one processor call and one dict per batch.
    for i in range(0, len(clips) - batch_size + 1, batch_size):
        batch_arrays = clips[i : i + batch_size]
        processed = processor(
            batch_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        # Convert BatchFeature (or any Mapping) to a plain dict of Tensors.
        yield {k: v for k, v in processed.items() if isinstance(v, torch.Tensor)}
