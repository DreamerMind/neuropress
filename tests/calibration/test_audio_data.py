"""Tests for load_librispeech_calibration (no network, no datasets install required)."""

import numpy as np
import pytest
import torch

from neuropress.calibration.data import load_librispeech_calibration

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

_SR = 16_000  # sampling rate used throughout


class _FakeProcessor:
    """Mimics a HuggingFace feature extractor returning {"input_values": Tensor}."""

    def __call__(
        self,
        audio_arrays,
        *,
        sampling_rate,
        return_tensors,
        padding=False,
    ):
        max_len = max(a.size for a in audio_arrays)
        padded = [
            np.pad(a, (0, max_len - a.size)) if a.size < max_len else a
            for a in audio_arrays
        ]
        tensor = torch.from_numpy(np.stack(padded))
        return {"input_values": tensor}


def _fake_dataset(n_clips: int = 200, clip_len_samples: int = 8_000):
    """Return a plain list of fake LibriSpeech examples with audio arrays."""
    rng = np.random.default_rng(0)
    return [
        {
            "audio": {
                "array": rng.random(clip_len_samples).astype(np.float32),
                "sampling_rate": _SR,
            }
        }
        for _ in range(n_clips)
    ]


def _variable_length_dataset(n_clips: int, min_len: int = 1000, max_len: int = 8000):
    """Return clips with varying lengths to test duration sort."""
    rng = np.random.default_rng(1)
    clips = []
    for i in range(n_clips):
        length = min_len + (i * (max_len - min_len)) // n_clips
        clips.append(
            {
                "audio": {
                    "array": rng.random(length).astype(np.float32),
                    "sampling_rate": _SR,
                }
            }
        )
    rng.shuffle(clips)
    return clips


def _loader(examples):
    """Wrap a list as a callable _load_dataset stub."""

    def _fn(*args, **kwargs):
        return iter(examples)

    return _fn


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------


def test_load_librispeech_returns_iterator_of_dicts():
    batches = load_librispeech_calibration(
        _FakeProcessor(),
        seq_len=8_000,
        n_samples=8,
        batch_size=2,
        _load_dataset=_loader(_fake_dataset()),
    )
    assert hasattr(batches, "__iter__") and hasattr(batches, "__next__")
    first = next(batches)
    assert isinstance(first, dict)
    assert len(first) > 0
    assert all(isinstance(v, torch.Tensor) for v in first.values())


def test_load_librispeech_batch_count():
    batches = load_librispeech_calibration(
        _FakeProcessor(),
        seq_len=8_000,
        n_samples=16,
        batch_size=4,
        _load_dataset=_loader(_fake_dataset()),
    )
    assert sum(1 for _ in batches) == 4  # 16 / 4


def test_load_librispeech_batch_size():
    first = next(
        iter(
            load_librispeech_calibration(
                _FakeProcessor(),
                seq_len=8_000,
                n_samples=8,
                batch_size=3,
                _load_dataset=_loader(_fake_dataset()),
            )
        )
    )
    tensor = next(iter(first.values()))
    assert tensor.shape[0] == 3


def test_load_librispeech_incomplete_last_batch_dropped():
    # n_samples=10, batch_size=3 → 3 full batches (9 samples), 1 dropped
    batches = list(
        load_librispeech_calibration(
            _FakeProcessor(),
            seq_len=8_000,
            n_samples=10,
            batch_size=3,
            _load_dataset=_loader(_fake_dataset()),
        )
    )
    assert len(batches) == 3
    for b in batches:
        tensor = next(iter(b.values()))
        assert tensor.shape[0] == 3


def test_load_librispeech_batch_size_one():
    batches = list(
        load_librispeech_calibration(
            _FakeProcessor(),
            seq_len=8_000,
            n_samples=5,
            batch_size=1,
            _load_dataset=_loader(_fake_dataset()),
        )
    )
    assert len(batches) == 5
    for b in batches:
        tensor = next(iter(b.values()))
        assert tensor.shape[0] == 1


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_seq_len_truncation_applied():
    """Clips longer than seq_len must be truncated before passing to processor."""
    seq_len = 4_000  # clips are 8_000 samples → should be halved

    class _LenCheckProcessor:
        """Asserts all arrays are at most seq_len long."""

        def __call__(self, arrays, *, sampling_rate, return_tensors, padding=False):
            for a in arrays:
                assert a.size <= seq_len, f"Expected ≤{seq_len}, got {a.size}"
            stacked = np.stack([np.pad(a, (0, seq_len - a.size)) for a in arrays])
            return {"input_values": torch.from_numpy(stacked)}

    list(
        load_librispeech_calibration(
            _LenCheckProcessor(),
            seq_len=seq_len,
            n_samples=8,
            batch_size=2,
            _load_dataset=_loader(_fake_dataset(clip_len_samples=8_000)),
        )
    )


# ---------------------------------------------------------------------------
# Duration sort
# ---------------------------------------------------------------------------


def test_clips_sorted_ascending_by_length():
    """Clips must be passed to processor in ascending length order."""
    seq_len = 8_000
    seen_lengths: list[int] = []

    class _SortCheckProcessor:
        def __call__(self, arrays, *, sampling_rate, return_tensors, padding=False):
            for a in arrays:
                seen_lengths.append(a.size)
            max_len = max(a.size for a in arrays)
            stacked = np.stack([np.pad(a, (0, max_len - a.size)) for a in arrays])
            return {"input_values": torch.from_numpy(stacked)}

    list(
        load_librispeech_calibration(
            _SortCheckProcessor(),
            seq_len=seq_len,
            n_samples=20,
            batch_size=1,
            _load_dataset=_loader(_variable_length_dataset(n_clips=30)),
        )
    )

    # Lengths seen by the processor must be non-decreasing.
    assert seen_lengths == sorted(seen_lengths), (
        f"Expected ascending order, got {seen_lengths[:5]}..."
    )


# ---------------------------------------------------------------------------
# Processor output passthrough
# ---------------------------------------------------------------------------


def test_processor_output_keys_are_forwarded():
    """All tensor keys returned by processor must appear in the yielded dicts."""

    class _MultiKeyProcessor:
        def __call__(self, arrays, *, sampling_rate, return_tensors, padding=False):
            t = torch.zeros(len(arrays), 10)
            return {"input_features": t, "attention_mask": torch.ones(len(arrays), 10)}

    first = next(
        iter(
            load_librispeech_calibration(
                _MultiKeyProcessor(),
                seq_len=8_000,
                n_samples=4,
                batch_size=2,
                _load_dataset=_loader(_fake_dataset()),
            )
        )
    )
    assert "input_features" in first
    assert "attention_mask" in first


def test_non_tensor_processor_outputs_are_excluded():
    """Non-tensor values returned by processor (e.g. metadata strings) are dropped."""

    class _MixedProcessor:
        def __call__(self, arrays, *, sampling_rate, return_tensors, padding=False):
            return {
                "input_values": torch.zeros(len(arrays), 8),
                "metadata": "not_a_tensor",
            }

    first = next(
        iter(
            load_librispeech_calibration(
                _MixedProcessor(),
                seq_len=8_000,
                n_samples=4,
                batch_size=2,
                _load_dataset=_loader(_fake_dataset()),
            )
        )
    )
    assert "input_values" in first
    assert "metadata" not in first


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_datasets_raises_import_error():
    def _raise(*a, **kw):
        raise ImportError("datasets")

    with pytest.raises(ImportError, match="datasets"):
        list(
            load_librispeech_calibration(
                _FakeProcessor(),
                seq_len=8_000,
                n_samples=4,
                _load_dataset=_raise,
            )
        )


def test_insufficient_clips_raises_runtime_error():
    tiny = _fake_dataset(n_clips=3)
    with pytest.raises(RuntimeError, match="n_samples"):
        list(
            load_librispeech_calibration(
                _FakeProcessor(),
                seq_len=8_000,
                n_samples=10,
                batch_size=1,
                _load_dataset=_loader(tiny),
            )
        )


def test_empty_audio_arrays_are_skipped():
    empty = {"audio": {"array": np.array([], dtype=np.float32), "sampling_rate": _SR}}
    examples = [empty] * 50
    examples += _fake_dataset(n_clips=50)
    batches = list(
        load_librispeech_calibration(
            _FakeProcessor(),
            seq_len=8_000,
            n_samples=8,
            batch_size=2,
            _load_dataset=_loader(examples),
        )
    )
    assert len(batches) == 4


def test_missing_audio_key_is_skipped():
    examples = [{"text": "hello world"}] * 50 + _fake_dataset(n_clips=50)
    batches = list(
        load_librispeech_calibration(
            _FakeProcessor(),
            seq_len=8_000,
            n_samples=8,
            batch_size=2,
            _load_dataset=_loader(examples),
        )
    )
    assert len(batches) == 4
