"""Calibration from a pre-tokenized safetensors file (``tokens:<path>``)."""

import pytest
import torch
from safetensors.torch import save_file

from neuropress.calibration.data import load_token_file_calibration
from neuropress.calibration.data.registry import resolve_calibration_loader


def _file(tmp_path, rows=6, cols=10):
    ids = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)
    path = tmp_path / "cal.safetensors"
    save_file({"input_ids": ids}, str(path))
    return path, ids


def test_yields_first_rows_cropped_and_batched(tmp_path):
    path, ids = _file(tmp_path)
    batches = list(
        load_token_file_calibration(
            None, path=str(path), seq_len=8, n_samples=5, batch_size=2, seed=0
        )
    )
    assert [b["input_ids"].shape for b in batches] == [(2, 8), (2, 8), (1, 8)]
    torch.testing.assert_close(
        torch.cat([b["input_ids"] for b in batches]), ids[:5, :8]
    )


@pytest.mark.parametrize(("seq_len", "n_samples"), [(11, 2), (8, 7)])
def test_rejects_requests_larger_than_the_file(tmp_path, seq_len, n_samples):
    path, _ = _file(tmp_path)
    with pytest.raises(ValueError, match=r"rows|columns"):
        list(
            load_token_file_calibration(
                None, path=str(path), seq_len=seq_len, n_samples=n_samples
            )
        )


def test_resolve_tokens_prefix_and_registry_names(tmp_path):
    path, ids = _file(tmp_path)
    loader = resolve_calibration_loader(f"tokens:{path}")
    (batch,) = list(loader(None, seq_len=4, n_samples=1, batch_size=1, seed=0))
    torch.testing.assert_close(batch["input_ids"], ids[:1, :4])

    from neuropress.calibration.data import load_c4_calibration

    assert resolve_calibration_loader("c4") is load_c4_calibration
    with pytest.raises(KeyError):
        resolve_calibration_loader("no_such_dataset")
