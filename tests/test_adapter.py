"""Tests for WeightAssignmentAdapter, particularly tied-weight safety.

Regression coverage for a bug found while investigating Qwen3.5-2B's VPTQ
collapse: quantizing a tied nn.Embedding (embed_tokens) via
MinMaxZpWeightEstimator silently overwrote the tied lm_head's weight too,
via ModTensorUpdater.update_by_ref's tie-preserving update (designed for
legitimate cases like "keep embed_tokens and lm_head in sync" but fatal
here since lm_head was never meant to receive this crude, un-Hessian-
weighted quantization at all). See neuropress/adapter.py and
torch_to_nnef/tensor/updater.py:_set_tensor for the mechanism.
"""

import torch
import torch.nn as nn

from neuropress.adapter import WeightAssignmentAdapter
from neuropress.estimators import MinMaxZpWeightEstimator


class _TiedModel(nn.Module):
    """Mirrors HF's tie_weights(): lm_head.weight IS embed_tokens.weight."""

    def __init__(self, vocab: int = 64, hidden: int = 16) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.embed_tokens(ids))


def test_quantizing_tied_embedding_does_not_corrupt_lm_head():
    torch.manual_seed(0)
    model = _TiedModel()
    orig_lm_head = model.lm_head.weight.data.clone()
    assert model.lm_head.weight is model.embed_tokens.weight  # sanity: tied

    est = MinMaxZpWeightEstimator(nbits=4, symmetric=True, weight_transform=None)
    est.prepare(model)
    plan = est.estimate(graph=None, registry=None, recipe=None)
    assert list(plan.weight_assignments) == ["embed_tokens"]

    WeightAssignmentAdapter().apply(model, plan)

    assert torch.equal(model.lm_head.weight.data, orig_lm_head), (
        "quantizing embed_tokens (never targeting lm_head) must not change "
        "lm_head's weight -- it did, meaning the tied reference was "
        "silently corrupted by the update mechanism."
    )
