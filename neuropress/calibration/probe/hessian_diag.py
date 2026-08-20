"""Hessian diagonal accumulator — lightweight importance-score collector.

Distinction from :class:`HessianAccumulator`
--------------------------------------------
:class:`HessianAccumulator` stores the *full* empirical Hessian matrix
``H`` of shape ``[K, K]`` where ``K`` is the number of input features.
This is required by GPTQ, which solves a Hessian-weighted least-squares
problem and needs the off-diagonal entries to propagate quantization error
column-by-column.

:class:`HessianDiagAccumulator` stores **only** ``diag(H)``, a vector of
shape ``[K]``.  It is 128x smaller for ``K = 4096`` and sufficient for
*importance-score* methods such as VPTQ that only need
``importance[row, col] ∝ diag(H)[col]`` — a per-column weight that is
broadcast uniformly across all output rows.

Mathematical derivation
-----------------------
The GPTQ-style empirical Hessian approximation is:

    ``H ≈ 2 * X @ X.T``  where ``X`` is ``[K, batch]`` (features × samples).

Its diagonal is therefore:

    ``diag(H)[j] = 2 * sum_t X[j, t]**2``

which equals ``2 * (X**2).sum(dim=1)``.  Accumulating this across batches
with a running weighted mean requires no matrix storage at all.

Usage::

    from neuropress.calibration.probe.hessian_diag import HessianDiagAccumulator

    acc = HessianDiagAccumulator()
    for batch in data_loader:
        acc.update(batch)
    diag_h = acc.result()["hessian_diag"]  # shape [K], float32
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from .collectors import Collector


@dataclass
class HessianDiagAccumulator(Collector):
    """Accumulates only the diagonal of the empirical Hessian approximation.

    Implements the :class:`~neuropress.calibration.probe.collectors.Collector`
    contract for the ``"hessian_diag"`` stat type.

    Unlike :class:`HessianAccumulator`, this collector stores a single
    ``[K]`` vector instead of a ``[K, K]`` matrix, making it practical for
    large feature dimensions.  The diagonal is sufficient for VPTQ-style
    importance weighting, where each column's importance is uniform across
    all output rows.

    The accumulation uses the same running-weighted-mean scheme as
    :class:`HessianAccumulator`:

        ``diag_H_new = diag_H * n/(n+b) + 2 * b/(n+b) * (scale * X)**2``

    so that the final result approximates ``2 * E[x_j**2]`` up to a
    ``nsamples`` scale factor consistent with GPTQ conventions.

    Attributes:
        hessian_diag: Running diagonal accumulation ``[K]``, float32.
            ``None`` until the first :meth:`update` call.
        nsamples: Total effective number of samples seen so far.
        observer_n_call: Number of :meth:`update` calls (for debugging).
    """

    hessian_diag: torch.Tensor | None = field(default=None, repr=False)
    nsamples: int = field(default=0)
    observer_n_call: int = field(default=0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset accumulation state to its initial (empty) values."""
        self.hessian_diag = None
        self.nsamples = 0
        self.observer_n_call = 0

    # ------------------------------------------------------------------
    # Collector contract
    # ------------------------------------------------------------------

    def update(self, data: torch.Tensor) -> None:
        """Accumulate the Hessian diagonal from one activation batch.

        Applies the same preprocessing as :class:`HessianAccumulator` for
        ``nn.Linear``-style inputs:

        - 2-D ``[batch, K]`` → unsqueeze → ``[1, batch, K]``
        - 3-D ``[batch, seq, K]`` → reshape to ``[batch*seq, K]``
        - Transpose to ``[K, n_eff]``

        Then accumulates ``diag(2 * X @ X.T) = 2 * (X**2).sum(dim=1)``
        via a running weighted mean.

        Args:
            data: Input activation tensor.  Any leading batch/seq dims are
                flattened; the last dim is taken as the feature dimension K.
        """
        self.observer_n_call += 1

        # Normalise to [K, n_eff] — same convention as HessianAccumulator.
        if data.dim() == 1:
            data = data.unsqueeze(0)
        if data.dim() == 3:
            data = data.reshape(-1, data.shape[-1])
        # data is now [n_eff, K]
        batch_size = data.shape[0]
        data = data.t().float()  # [K, n_eff]

        n_total = self.nsamples + batch_size
        scale = math.sqrt(2.0 * batch_size / n_total)
        contrib = (scale * data).pow(2).sum(dim=1)  # [K]

        if self.hessian_diag is None:
            self.hessian_diag = contrib
        else:
            self.hessian_diag = self.hessian_diag * (self.nsamples / n_total) + contrib

        self.nsamples = n_total

    def result(self) -> dict[str, Any]:
        """Return accumulated statistics.

        Returns:
            Dict with keys ``"hessian_diag"`` (``torch.Tensor | None``,
            shape ``[K]``) and ``"nsamples"`` (``int``).
        """
        return {"hessian_diag": self.hessian_diag, "nsamples": self.nsamples}
