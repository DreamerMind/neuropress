"""The Hessian-weighted M-step must weight every group, not just the first one."""

import torch

from neuropress.estimators import VPTQGroupEstimator
from neuropress.transforms.vq_packing import IndexPacking


def _layer(rows: list[list[float]]) -> torch.nn.Linear:
    w = torch.tensor(rows)
    mod = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
    mod.weight = torch.nn.Parameter(w)
    return mod


def _refine_one_centroid(est, name, mod, weights):
    """Assign every vector of *mod* to centroid 0 and run one M-step."""
    est._compute_layer_vector_weights = lambda n, m, reg, column_perm=None: weights[n]
    est._layer_has_hessian = lambda n, m, reg: True
    centroids = torch.zeros(16, 2, dtype=torch.float16)
    n_vec = mod.weight.numel() // 2
    packed = {name: IndexPacking.pack(torch.zeros(n_vec, dtype=torch.long), 4)}
    est._refine_centroids([(name, mod)], packed, centroids, {}, registry=object())
    return centroids[0].float()


def test_second_group_m_step_is_hessian_weighted():
    est = VPTQGroupEstimator(
        module_classes=(torch.nn.Linear,),
        num_centroids=16,
        vector_len=2,
        centroid_weighting="hessian",
    )
    est._device = torch.device("cpu")
    # Two vectors per layer (a 4-bit index each), [0, 0] and [4, 4]; weight 3 on the second one.
    weights = {
        "a": torch.tensor([1.0, 3.0]),
        "b": torch.tensor([1.0, 3.0]),
    }
    first = _refine_one_centroid(est, "a", _layer([[0.0, 0.0], [4.0, 4.0]]), weights)
    second = _refine_one_centroid(est, "b", _layer([[0.0, 0.0], [4.0, 4.0]]), weights)
    expected = torch.tensor([3.0, 3.0])  # (1*0 + 3*4) / 4, not the plain mean 2
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
