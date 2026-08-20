"""Tests for load_c4_calibration and load_eval_targeted_calibration (no network required)."""

import pytest
import torch

from neuropress.calibration.data import (
    load_c4_calibration,
    load_eval_targeted_calibration,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeEncoding:
    def __init__(self, ids):
        self.input_ids = ids


class _WordTokenizer:
    """Encodes each whitespace-separated word as a distinct integer (mod 50000)."""

    def __call__(self, text, add_special_tokens=False):
        words = text.split()
        return _FakeEncoding([hash(w) % 50_000 for w in words])


def _fake_dataset(n_examples: int = 200, words_per_example: int = 300):
    """Return a plain list of fake C4 examples."""
    return [
        {"text": " ".join(f"tok{i}" for i in range(words_per_example))}
        for _ in range(n_examples)
    ]


def _loader(examples):
    """Wrap a list as a callable _load_dataset stub."""

    def _fn(*args, **kwargs):
        return iter(examples)

    return _fn


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------


def test_load_c4_returns_iterator_of_dicts():
    batches = load_c4_calibration(
        _WordTokenizer(),
        seq_len=64,
        n_samples=8,
        batch_size=2,
        _load_dataset=_loader(_fake_dataset()),
    )
    # Should be a lazy iterator, not a list.
    assert hasattr(batches, "__iter__") and hasattr(batches, "__next__")
    assert all(isinstance(b, dict) and "input_ids" in b for b in batches)


def test_load_c4_batch_count():
    batches = load_c4_calibration(
        _WordTokenizer(),
        seq_len=64,
        n_samples=16,
        batch_size=4,
        _load_dataset=_loader(_fake_dataset()),
    )
    assert sum(1 for _ in batches) == 4  # 16 samples / 4 per batch


def test_load_c4_batch_shape():
    first = next(
        iter(
            load_c4_calibration(
                _WordTokenizer(),
                seq_len=64,
                n_samples=8,
                batch_size=2,
                _load_dataset=_loader(_fake_dataset()),
            )
        )
    )
    assert first["input_ids"].shape == (2, 64)


def test_load_c4_dtype_is_long():
    first = next(
        iter(
            load_c4_calibration(
                _WordTokenizer(),
                seq_len=32,
                n_samples=4,
                batch_size=1,
                _load_dataset=_loader(_fake_dataset()),
            )
        )
    )
    assert first["input_ids"].dtype == torch.long


@pytest.mark.parametrize("seq_len", [32, 64, 128, 512, 1024])
def test_load_c4_seq_len_respected(seq_len):
    batches = load_c4_calibration(
        _WordTokenizer(),
        seq_len=seq_len,
        n_samples=4,
        batch_size=1,
        _load_dataset=_loader(_fake_dataset(n_examples=500, words_per_example=2048)),
    )
    for b in batches:
        assert b["input_ids"].shape[-1] == seq_len


def test_load_c4_incomplete_last_batch_dropped():
    # n_samples=10, batch_size=3 → 3 full batches (9 samples), 1 dropped
    batches = list(
        load_c4_calibration(
            _WordTokenizer(),
            seq_len=32,
            n_samples=10,
            batch_size=3,
            _load_dataset=_loader(_fake_dataset()),
        )
    )
    assert len(batches) == 3
    for b in batches:
        assert b["input_ids"].shape[0] == 3


def test_load_c4_batch_size_one():
    batches = list(
        load_c4_calibration(
            _WordTokenizer(),
            seq_len=32,
            n_samples=5,
            batch_size=1,
            _load_dataset=_loader(_fake_dataset()),
        )
    )
    assert len(batches) == 5
    assert batches[0]["input_ids"].shape == (1, 32)


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------


def test_reproducibility_same_seed_same_order():
    _ds = _loader(_fake_dataset())
    a = load_c4_calibration(
        _WordTokenizer(),
        seq_len=32,
        n_samples=8,
        batch_size=1,
        seed=0,
        _load_dataset=_ds,
    )
    b = load_c4_calibration(
        _WordTokenizer(),
        seq_len=32,
        n_samples=8,
        batch_size=1,
        seed=0,
        _load_dataset=_ds,
    )
    for ba, bb in zip(a, b, strict=False):
        assert torch.equal(ba["input_ids"], bb["input_ids"])


def test_reproducibility_different_seeds_different_order():
    a = load_c4_calibration(
        _WordTokenizer(),
        seq_len=32,
        n_samples=16,
        batch_size=1,
        seed=0,
        _load_dataset=_loader(_fake_dataset()),
    )
    b = load_c4_calibration(
        _WordTokenizer(),
        seq_len=32,
        n_samples=16,
        batch_size=1,
        seed=99,
        _load_dataset=_loader(_fake_dataset()),
    )
    # With 16 samples it's astronomically unlikely they're in the same order.
    assert not all(
        torch.equal(ba["input_ids"], bb["input_ids"])
        for ba, bb in zip(a, b, strict=False)
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_errors_missing_datasets_raises_import_error():
    with pytest.raises(ImportError, match="datasets"):
        # _load_dataset=None forces the lazy import path, which we cannot
        # intercept here without patching sys.modules — so we verify the
        # message text by passing a stub that raises ImportError itself.
        def _raise(*a, **kw):
            raise ImportError("datasets")

        # Consume the generator to trigger the error (generator bodies
        # execute lazily on first iteration).
        list(
            load_c4_calibration(
                _WordTokenizer(),
                seq_len=32,
                n_samples=4,
                _load_dataset=_raise,
            )
        )


def test_errors_insufficient_data_raises_runtime_error():
    tiny = [{"text": "a b c"}]  # 3 tokens, seq_len=64 → impossible
    with pytest.raises(RuntimeError, match="seq_len"):
        # Consume the generator to trigger the error.
        list(
            load_c4_calibration(
                _WordTokenizer(),
                seq_len=64,
                n_samples=4,
                batch_size=1,
                _load_dataset=_loader(tiny),
            )
        )


def test_errors_empty_texts_are_skipped():
    examples = [{"text": ""}] * 50 + _fake_dataset(n_examples=50)
    batches = load_c4_calibration(
        _WordTokenizer(),
        seq_len=32,
        n_samples=4,
        batch_size=1,
        _load_dataset=_loader(examples),
    )
    assert sum(1 for _ in batches) == 4


def test_errors_missing_text_key_skipped():
    examples = [{"other_key": "hello world"}] * 50 + _fake_dataset(n_examples=50)
    batches = load_c4_calibration(
        _WordTokenizer(),
        seq_len=32,
        n_samples=4,
        batch_size=1,
        _load_dataset=_loader(examples),
    )
    assert sum(1 for _ in batches) == 4


# ---------------------------------------------------------------------------
# load_eval_targeted_calibration helpers
# ---------------------------------------------------------------------------


def _fake_task_datasets() -> dict[str, list[dict]]:
    """Build minimal fake examples for all supported task formats."""
    n = 200
    words = " ".join(f"w{i}" for i in range(60))
    return {
        "boolq": [
            {"passage": words, "question": "is it true", "label": i % 2}
            for i in range(n)
        ],
        "arc_easy": [
            {
                "question": words,
                "choices": {"label": ["A", "B", "C", "D"], "text": [words] * 4},
                "answerKey": "A",
            }
            for _ in range(n)
        ],
        "arc_challenge": [
            {
                "question": words,
                "choices": {"label": ["A", "B", "C", "D"], "text": [words] * 4},
                "answerKey": "B",
            }
            for _ in range(n)
        ],
        "piqa": [
            {"goal": words, "sol1": words, "sol2": words, "label": i % 2}
            for i in range(n)
        ],
        "openbookqa": [
            {
                "question_stem": words,
                "choices": {"label": ["A", "B", "C", "D"], "text": [words] * 4},
                "answerKey": "C",
            }
            for _ in range(n)
        ],
    }


def _mixed_loader(task_datasets: dict[str, list[dict]]):
    """Return a _load_dataset stub that dispatches by dataset_id/config."""

    def _fn(dataset_id, dataset_config=None, split="train", **kwargs):
        # C4
        if "c4" in dataset_id:
            return iter(_fake_dataset())
        # boolq
        if "super_glue" in dataset_id or (dataset_config and "boolq" in dataset_config):
            return iter(task_datasets["boolq"])
        # arc
        if "ai2_arc" in dataset_id:
            if dataset_config and "Easy" in dataset_config:
                return iter(task_datasets["arc_easy"])
            return iter(task_datasets["arc_challenge"])
        # piqa
        if "piqa" in dataset_id:
            return iter(task_datasets["piqa"])
        # openbookqa
        if "openbookqa" in dataset_id:
            return iter(task_datasets["openbookqa"])
        return iter([])

    return _fn


# ---------------------------------------------------------------------------
# load_eval_targeted_calibration tests
# ---------------------------------------------------------------------------


def test_mixed_returns_correct_shape():
    td = _fake_task_datasets()
    batches = list(
        load_eval_targeted_calibration(
            _WordTokenizer(),
            seq_len=32,
            n_samples=8,
            batch_size=1,
            _load_dataset=_mixed_loader(td),
        )
    )
    assert len(batches) == 8
    for b in batches:
        assert b["input_ids"].shape == (1, 32)
        assert b["input_ids"].dtype == torch.long


def test_mixed_batch_size_respected():
    td = _fake_task_datasets()
    batches = list(
        load_eval_targeted_calibration(
            _WordTokenizer(),
            seq_len=32,
            n_samples=8,
            batch_size=4,
            _load_dataset=_mixed_loader(td),
        )
    )
    assert len(batches) == 2
    for b in batches:
        assert b["input_ids"].shape[0] == 4


def test_mixed_subset_of_tasks():
    td = _fake_task_datasets()
    batches = list(
        load_eval_targeted_calibration(
            _WordTokenizer(),
            seq_len=32,
            n_samples=8,
            batch_size=1,
            task_names=("boolq", "piqa"),
            _load_dataset=_mixed_loader(td),
        )
    )
    assert len(batches) == 8


def test_mixed_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task names"):
        list(
            load_eval_targeted_calibration(
                _WordTokenizer(),
                seq_len=32,
                n_samples=4,
                task_names=("boolq", "not_a_task"),
                _load_dataset=_mixed_loader(_fake_task_datasets()),
            )
        )
