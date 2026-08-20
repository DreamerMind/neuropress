"""Tests for selecting a clustering backend by name from a recipe config."""

import pytest

import neuropress.t2n.qtensors  # noqa: F401
from neuropress.transforms.cluster_backends import (
    IVFTorchKMeansBackend,
    TorchKMeansBackend,
    backend_by_name,
)

# ---------------------------------------------------------------------------
# The auto-selector routes n_clusters > 4096 to IVFTorchKMeansBackend, whose
# torch.compile'd kernel materialises a [N, nlist*cell_size] slab ~45x wider
# than its own chunk budget assumes - which OOM'd both a 24 GiB and a 40 GiB
# card on 2026-08-04.  TorchKMeansBackend chunks correctly, so it must be
# selectable from a recipe without dropping NC (which would change bpw).
# ---------------------------------------------------------------------------


def test_selects_chunked_torch_backend():
    assert isinstance(backend_by_name("torch"), TorchKMeansBackend)


def test_selects_ivf_backend():
    assert isinstance(backend_by_name("ivf"), IVFTorchKMeansBackend)


def test_auto_returns_none_so_caller_uses_autoselect():
    assert backend_by_name("auto") is None
    assert backend_by_name(None) is None


def test_forwards_seed_and_iterations():
    b = backend_by_name("torch", seed=7, n_iter=80)
    assert b.seed == 7
    assert b.n_iter == 80


def test_unknown_name_raises_with_valid_options():
    with pytest.raises(ValueError, match="torch"):
        backend_by_name("nope")
