"""Paired observations for linear teacher-output reconstruction."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .collectors import PairedCollector


@dataclass
class LinearReconstructionCollector(PairedCollector):
    """Accumulate teacher residual/input products with Hessian-compatible weighting.

    The residual is teacher output minus student output before quantization.
    Bias stays fixed. Inputs and outputs must have aligned leading dimensions.
    """

    module: nn.Module | None = None
    cross: torch.Tensor | None = None
    nsamples: int = 0

    def update_pair(self, inputs: tuple, output: Any, reference: Any) -> None:
        """Accumulate aligned linear input and teacher residual products."""
        if not isinstance(self.module, nn.Linear):
            raise ValueError("Linear reconstruction requires nn.Linear targets")
        x = inputs[0]
        if not isinstance(output, torch.Tensor) or not isinstance(
            reference, torch.Tensor
        ):
            raise ValueError("Linear reconstruction requires tensor outputs")
        if output.shape != reference.shape or x.shape[:-1] != output.shape[:-1]:
            raise ValueError("Teacher and student observations must be aligned")
        batch_size = x.shape[0] if x.ndim > 2 else 1
        total = self.nsamples + batch_size
        residual = (reference.to(output.device).float() - output.float()).reshape(
            -1, output.shape[-1]
        )
        contribution = residual.t() @ x.float().reshape(-1, x.shape[-1])
        if self.cross is None:
            self.cross = torch.zeros_like(contribution)
        self.cross = self.cross * (self.nsamples / total) + contribution * (
            2 * batch_size / total
        )
        self.nsamples = total

    def result(self) -> dict[str, Any]:
        """Return residual products and the normalization sample count."""
        return {"cross": self.cross, "nsamples": self.nsamples}


@torch.no_grad()
def reconstruct_linear_weight(
    weight: torch.Tensor, hessian: torch.Tensor, cross: torch.Tensor, damping: float
) -> torch.Tensor:
    """Fit a ridge-regularized teacher correction without changing the source weight.

    Minimizes teacher-output squared error plus lambda * ||W_new - W||².
    ``cross`` is the teacher-minus-student output/input product; it must use
    the same normalization as ``hessian``. Lambda is damping times mean(H.diag).
    """
    if damping <= 0:
        raise ValueError("Reconstruction damping must be positive")
    h = hessian.to(device=weight.device, dtype=torch.float32).clone()
    c = cross.to(device=weight.device, dtype=torch.float32)
    if h.shape != (weight.shape[1], weight.shape[1]) or c.shape != weight.shape:
        raise ValueError("Reconstruction statistic shapes do not match the weight")
    ridge = damping * h.diag().mean().clamp_min(torch.finfo(h.dtype).eps)
    h.diagonal().add_(ridge)
    factor = torch.linalg.cholesky(h)
    correction = torch.cholesky_solve(c.t(), factor).t()
    result = weight.float() + correction
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite reconstructed weight")
    return result.to(weight.dtype)
