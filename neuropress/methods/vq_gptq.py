"""Post-convergence GPTQ sequential correction for VQ assignments.

After Lloyd's iterations converge, :class:`VQGPTQCorrector` refines the index
assignment by propagating quantization error left-to-right through the weight
matrix using the Cholesky inverse of the input Hessian — exactly the GPTQ
update, applied to VQ column groups instead of scalar quantization columns.

The update for group ``g`` spanning columns ``[c0, c1)`` is:

    E_norm = (W[:, c0:c1] - C[idx_g]) / diag(U[c0:c1, c0:c1])
    W[:, c1:] -= alpha * E_norm @ U[c0:c1, c1:]

where ``U`` is the upper Cholesky factor of ``H^{-1}`` and ``alpha`` is
:attr:`VQGPTQCorrector.gptq_correction_alpha`.

With ``block_metric=True`` the update is exact block LDLQ instead: the block's
own triangle ``U_bb = U[c0:c1, c0:c1]`` defines both the assignment metric and
the feedback,

    Z = (W'[:, c0:c1] - C[idx_g]) @ inv(U_bb)      idx_g = argmin ||(W'_b - c) inv(U_bb)||
    W'[:, c1:] -= Z @ U[c0:c1, c1:]

and the sum of ``||Z||^2`` over blocks equals ``tr((W - Q) H (W - Q)^T)``
exactly.  :class:`LDLQStats` accumulates that loss as a quadratic in the
codebook, so an M-step can re-fit the centroids to the full-Hessian objective.

This module is purely computational — callers are responsible for loading
the full ``[K, K]`` Hessian and the optional diagonal importance weights
before calling :meth:`VQGPTQCorrector.correct`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)

# Device types already warned about. Process-wide, because block calibration
# builds a fresh estimator, and so a fresh corrector, for every stage.
_WARNED_DEVICE_TYPES: set[str] = set()


@dataclass
class LDLQStats:
    """The block-LDLQ loss of one codebook as a quadratic in its centroids.

    For fixed assignments and fixed error-fed targets ``w'`` (recorded during a
    block-LDLQ pass), the loss is
    ``sum_k c_k^T S_k c_k - 2 c_k . T_k + const`` with
    ``S_k = sum A_b`` and ``T_k = sum w'_rb A_b`` over the vectors assigned to
    centroid ``k``, where ``A_b = inv(U_bb) inv(U_bb)^T``.  Sums stay on the
    device (float64) so accumulating does not synchronize.
    """

    S: torch.Tensor  # [NC, VL, VL]
    T: torch.Tensor  # [NC, VL]
    const_acc: torch.Tensor  # scalar
    loss_acc: torch.Tensor  # scalar: loss of the assignments the pass made
    n_layers: int = 0

    @classmethod
    def empty(cls, num_centroids: int, vector_len: int) -> LDLQStats:
        """Return zeroed statistics for a ``[num_centroids, vector_len]`` codebook."""
        z = torch.zeros((), dtype=torch.float64)
        return cls(
            S=torch.zeros(num_centroids, vector_len, vector_len, dtype=torch.float64),
            T=torch.zeros(num_centroids, vector_len, dtype=torch.float64),
            const_acc=z,
            loss_acc=z.clone(),
        )

    @property
    def loss(self) -> float:
        """Loss of the assignments made, ``tr((W - Q) H (W - Q)^T)`` summed."""
        return float(self.loss_acc)

    @property
    def const(self) -> float:
        """The codebook-independent term of :meth:`loss_for`."""
        return float(self.const_acc)

    def _to(self, dev: torch.device) -> None:
        if self.S.device != dev:
            self.S, self.T = self.S.to(dev), self.T.to(dev)
            self.const_acc = self.const_acc.to(dev)
            self.loss_acc = self.loss_acc.to(dev)

    def add_block(
        self, w_block: torch.Tensor, idx: torch.Tensor, a: torch.Tensor, z: torch.Tensor
    ) -> None:
        """Accumulate one VL-column block.

        Args:
            w_block: Error-fed targets ``w'`` ``[R, VL]``.
            idx: Assigned centroid per row ``[R]``.
            a: The block metric ``A_b`` ``[VL, VL]``.
            z: The normalized residual ``(w' - q) inv(U_bb)`` ``[R, VL]``.
        """
        self._to(w_block.device)
        w = w_block.double()
        a = a.double()
        y = w @ a
        vl = a.shape[0]
        self.S.view(-1, vl * vl).index_add_(
            0, idx, a.reshape(1, vl * vl).expand(idx.shape[0], -1)
        )
        self.T.index_add_(0, idx, y)
        self.const_acc += (y * w).sum()
        self.loss_acc += z.double().square().sum()

    def loss_for(self, centroids: torch.Tensor) -> float:
        """Return the accumulated loss if the codebook were ``centroids``."""
        c = centroids.to(self.S.device, torch.float64)
        quad = torch.einsum("ki,kij,kj->", c, self.S, c)
        return float(self.const_acc - 2.0 * (c * self.T).sum() + quad)

    def solve(self, centroids: torch.Tensor) -> torch.Tensor:
        """Return the loss-minimizing codebook; unused centroids are kept as-is."""
        c_old = centroids.to(self.S.device, torch.float64)
        used = self.S.diagonal(dim1=1, dim2=2).sum(-1) > 0
        out = c_old.clone()
        if used.any():
            s = self.S[used]
            ridge = 1e-9 * s.diagonal(dim1=1, dim2=2).mean(-1).clamp(min=1e-30)
            eye = torch.eye(s.shape[-1], dtype=s.dtype, device=s.device)
            s = s + ridge[:, None, None] * eye
            out[used] = torch.linalg.solve(s, self.T[used].unsqueeze(-1)).squeeze(-1)
        return out.to(centroids.device, centroids.dtype)


@dataclass
class VQGPTQCorrector:
    """GPTQ sequential error-propagation corrector for VQ index assignment.

    Instantiate once per estimator (or per group) and call :meth:`correct`
    for each layer after Lloyd's iterations have converged.

    Args:
        vector_len: VQ vector length — number of consecutive weight columns
            per group (must match the codebook).
        gptq_correction_alpha: Scale factor applied to the GPTQ error
            propagation delta.  ``1.0`` = full GPTQ correction.
        gptq_damp_percent: Initial fractional dampening applied to the
            Hessian diagonal before Cholesky decomposition.
        gptq_damp_auto_increment: Per-retry increment added to
            ``gptq_damp_percent`` when Cholesky fails.
        centroid_attraction: Strength of the look-ahead centroid-attraction
            incentive.  ``0.0`` disables it.
        centroid_attraction_window: Number of future groups nudged toward
            their nearest centroid after each GPTQ propagation step.
        block_metric: Exact block LDLQ (see the module docstring): assign
            and propagate with the full ``U_bb`` block instead of its
            diagonal, ignoring ``diag_h``.  Also runs on CPU.
    """

    vector_len: int
    gptq_correction_alpha: float = 1.0
    gptq_damp_percent: float = 0.01
    gptq_damp_auto_increment: float = 0.0025
    centroid_attraction: float = 0.0
    centroid_attraction_window: int = 4
    block_metric: bool = False

    def __post_init__(self) -> None:
        if self.block_metric and (
            self.centroid_attraction or self.gptq_correction_alpha != 1.0
        ):
            raise ValueError(
                "block_metric is exact LDLQ: centroid_attraction must be 0 and "
                "gptq_correction_alpha 1."
            )

    def correct(
        self,
        name: str,
        mod: nn.Module,
        centroids: torch.Tensor,
        full_h: torch.Tensor,
        device: torch.device,
        column_perm: torch.Tensor | None = None,
        diag_h: torch.Tensor | None = None,
        mstep_stats: LDLQStats | None = None,
    ) -> torch.Tensor | None:
        """Run GPTQ-corrected VQ assignment for one layer.

        See :meth:`correct_residual` for the two-stage variant.
        """
        out = self._correct(
            name, mod, centroids, full_h, device, column_perm, diag_h, mstep_stats
        )
        return None if out is None else out[0]

    def correct_residual(
        self,
        name: str,
        mod: nn.Module,
        centroids: torch.Tensor,
        residual_centroids: torch.Tensor,
        residual_groups: int,
        full_h: torch.Tensor,
        device: torch.device,
        column_perm: torch.Tensor | None = None,
        diag_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """GPTQ-corrected assignment with a residual stage on the first groups.

        The first ``residual_groups`` column groups (in ``column_perm`` order)
        are coded as ``centroids[i] + residual_centroids[j]`` by
        :func:`~neuropress.methods.residual_vq.two_stage_assign`, and their
        two-stage error is what feeds forward. Not available with
        ``block_metric``.

        Returns:
            ``(packed base indices, residual indices [rows, residual_groups])``,
            or ``None`` when correction is infeasible.
        """
        if self.block_metric:
            raise ValueError("Residual VQ is implemented for the diagonal metric only")
        out = self._correct(
            name,
            mod,
            centroids,
            full_h,
            device,
            column_perm,
            diag_h,
            None,
            residual=(residual_centroids, residual_groups),
        )
        if out is None:
            return None
        assert out[1] is not None
        return out[0], out[1]

    def _correct(
        self,
        name: str,
        mod: nn.Module,
        centroids: torch.Tensor,
        full_h: torch.Tensor,
        device: torch.device,
        column_perm: torch.Tensor | None,
        diag_h: torch.Tensor | None,
        mstep_stats: LDLQStats | None,
        residual: tuple[torch.Tensor, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """Shared body of :meth:`correct` and :meth:`correct_residual`.

        Args:
            name: Layer name (used for logging).
            mod: The module whose ``weight`` will be reassigned.
            centroids: Shared codebook ``[NC, VL]``.
            full_h: Full input Hessian ``[K, K]`` for this layer.
            device: Target CUDA device.
            column_perm: Optional Hessian-sorted column permutation ``[K]``.
            diag_h: Optional Hessian diagonal ``[K]`` for importance-weighted
                assignment within each group.  Falls back to uniform if None.
            mstep_stats: With ``block_metric``, accumulates this layer's loss
                as a function of the codebook (see :class:`LDLQStats`).
            residual: Optional ``(residual codebook, groups)``; see
                :meth:`correct_residual`.

        Returns:
            ``(packed base indices, residual indices or None)``; the packed
            tensor has the format of
            :func:`~neuropress.transforms.vq_packing.IndexPacking.pack`.
            ``None`` when correction is infeasible (VRAM too tight, Cholesky
            failure, etc.); the caller then keeps the plain assignment.
        """
        from ..kernels import kmeans_assign as _kmeans_assign
        from ..transforms.vq_packing import IndexPacking

        w = mod.weight.data
        if w.ndim < 2:
            return None
        K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
        if K % self.vector_len != 0:
            return None

        VL = self.vector_len
        bits = IndexPacking.bits_for(centroids.shape[0])

        if device.type != "cuda" and not self.block_metric:
            if device.type not in _WARNED_DEVICE_TYPES:
                _WARNED_DEVICE_TYPES.add(device.type)
                LOGGER.warning(
                    "VQGPTQCorrector: GPTQ correction needs a CUDA estimation "
                    "device, got %s; keeping plain VQ assignments.",
                    device,
                )
            return None
        w_fp32_bytes = w.numel() * 4
        free_bytes = (
            torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else None
        )
        if free_bytes is not None and w_fp32_bytes > free_bytes * 0.4:
            LOGGER.info(
                "VQGPTQCorrector: skip '%s' - fp32 weight (%d MiB) exceeds VRAM budget.",
                name,
                w_fp32_bytes >> 20,
            )
            return None

        R = w.shape[0]
        w2d = w.float().to(device).reshape(R, K)
        H = full_h.float().to(device)

        if column_perm is not None:
            perm = column_perm.to(device)
            w2d = w2d[:, perm]
            H = H[perm][:, perm]
        # Without a permutation, w2d is the module weight itself when that is
        # already float32 on the device; every write below goes to this copy.
        w_corr = w2d.clone()
        del w2d

        dead = H.diagonal() == 0
        if dead.any():
            if H is full_h:
                # Already float32 on the device, so H is the caller's (often
                # a registry statistic); patch a copy instead.
                H = H.clone()
            H.diagonal()[dead] = 1.0
            w_corr[:, dead] = 0.0

        try:
            U = self.cholesky_inv_upper(H)
        except Exception as exc:
            LOGGER.warning(
                "VQGPTQCorrector: Cholesky failed for '%s': %s - keeping plain assignment.",
                name,
                exc,
            )
            return None
        del H

        if self.block_metric:
            # w_corr is already this call's own copy, dead columns zeroed.
            indices = self._block_ldlq(
                w_corr, U, centroids.float().to(device), mstep_stats
            )
            return IndexPacking.pack(indices, bits), None

        if diag_h is not None:
            diag_h = diag_h.float().to(device)
            if column_perm is not None:
                diag_h = diag_h[column_perm.to(device)]

        C = centroids.float().to(device)
        NC = C.shape[0]
        G = K // VL
        U_diag = U.diagonal()

        all_indices: list[torch.Tensor] = []
        res_indices: list[torch.Tensor] = []
        RC, n_res = (None, 0) if residual is None else residual
        if RC is not None:
            from .residual_vq import two_stage_assign

            RC = RC.float().to(device)

        do_attraction = self.centroid_attraction > 0.0
        U_diag_sq_mean = (U_diag**2).mean().item() if do_attraction else 1.0

        for g in range(G):
            c0, c1 = g * VL, (g + 1) * VL
            w_block = w_corr[:, c0:c1]

            if diag_h is not None:
                imp = diag_h[c0:c1].clamp(min=1e-8).sqrt()
            else:
                imp = torch.ones(VL, device=device)
            w_scaled = w_block * imp.unsqueeze(0)
            c_scaled = C * imp.unsqueeze(0)

            if g < n_res:
                assert RC is not None
                idx_g, ridx_g = two_stage_assign(
                    w_scaled, c_scaled, RC * imp.unsqueeze(0)
                )
                res_indices.append(ridx_g)
            elif VL == 4:
                idx_g = _kmeans_assign(w_scaled.contiguous(), c_scaled.contiguous(), VL)
            else:
                free, _ = torch.cuda.mem_get_info(device)
                chunk = max(1, min(R, int(free * 0.3) // (NC * 4)))
                if chunk >= R:
                    idx_g = torch.cdist(w_scaled, c_scaled).argmin(dim=1)
                else:
                    parts = []
                    for s in range(0, R, chunk):
                        parts.append(
                            torch.cdist(w_scaled[s : s + chunk], c_scaled).argmin(dim=1)
                        )
                    idx_g = torch.cat(parts)
            all_indices.append(idx_g)

            if c1 < K:
                E = w_block - C[idx_g]
                if g < n_res:
                    E = E - RC[res_indices[-1]]
                E_norm = E / U_diag[c0:c1].unsqueeze(0)
                delta = E_norm @ U[c0:c1, c1:]
                if g == 0:
                    LOGGER.info(
                        "VQGPTQCorrector '%s': U_diag[0:VL] mean=%.4e  E mean=%.4e  "
                        "delta mean=%.4e  weight mean=%.4e  ratio=%.2f",
                        name,
                        U_diag[c0:c1].abs().mean().item(),
                        E.abs().mean().item(),
                        delta.abs().mean().item(),
                        w_corr.abs().mean().item(),
                        delta.abs().mean().item() / (w_corr.abs().mean().item() + 1e-9),
                    )
                w_corr[:, c1:] -= self.gptq_correction_alpha * delta

                if do_attraction:
                    win_end = min(G, g + 1 + self.centroid_attraction_window)
                    for g2 in range(g + 1, win_end):
                        c0_g2, c1_g2 = g2 * VL, (g2 + 1) * VL
                        w_g2 = w_corr[:, c0_g2:c1_g2].contiguous()
                        if VL == 4:
                            snap_idx = _kmeans_assign(w_g2, C, VL)
                        else:
                            free2, _ = torch.cuda.mem_get_info(device)
                            chunk2 = max(1, min(R, int(free2 * 0.3) // (NC * 4)))
                            if chunk2 >= R:
                                snap_idx = torch.cdist(w_g2, C).argmin(dim=1)
                            else:
                                idx_parts = []
                                for s in range(0, R, chunk2):
                                    idx_parts.append(
                                        torch.cdist(w_g2[s : s + chunk2], C).argmin(
                                            dim=1
                                        )
                                    )
                                snap_idx = torch.cat(idx_parts)
                        snap = C[snap_idx] - w_g2
                        h_scale = (
                            U_diag[c0_g2:c1_g2] ** 2
                        ).mean().item() / U_diag_sq_mean
                        w_corr[:, c0_g2:c1_g2] = (
                            w_g2 + self.centroid_attraction * h_scale * snap
                        )

        indices = torch.stack(all_indices, dim=1).reshape(-1).to(torch.int64)
        res = torch.stack(res_indices, dim=1).to(torch.int64) if res_indices else None
        if residual is not None and res is None:
            res = torch.empty(R, 0, dtype=torch.int64, device=device)
        return IndexPacking.pack(indices, bits), res

    def _block_ldlq(
        self,
        w: torch.Tensor,
        U: torch.Tensor,
        C: torch.Tensor,
        stats: LDLQStats | None,
        lazy_cols: int = 128,
    ) -> torch.Tensor:
        """Exact block LDLQ over VL-column blocks; returns flat row-major indices.

        Feedback is applied lazily as in GPTQ: inside a chunk of ``lazy_cols``
        columns, then to the rest of the matrix once per chunk.  The result is
        the same as updating every remaining column after every block.
        """
        VL = self.vector_len
        K = w.shape[1]
        step = max(VL, lazy_cols - lazy_cols % VL)
        eye = torch.eye(VL, device=w.device, dtype=w.dtype)
        idx_cols: list[torch.Tensor] = []
        for b0 in range(0, K, step):
            b1 = min(K, b0 + step)
            zs: list[torch.Tensor] = []
            for c0 in range(b0, b1, VL):
                c1 = c0 + VL
                m = torch.linalg.solve_triangular(U[c0:c1, c0:c1], eye, upper=True)
                w_block = w[:, c0:c1]
                idx = self._nearest(w_block @ m, C @ m)
                idx_cols.append(idx)
                z = (w_block - C[idx]) @ m
                if stats is not None:
                    stats.add_block(w_block, idx, m @ m.T, z)
                if c1 < b1:
                    w[:, c1:b1] -= z @ U[c0:c1, c1:b1]
                zs.append(z)
            if b1 < K:
                w[:, b1:] -= torch.cat(zs, dim=1) @ U[b0:b1, b1:]
        if stats is not None:
            stats.n_layers += 1
        return torch.stack(idx_cols, dim=1).reshape(-1).to(torch.int64)

    def _nearest(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Index of the nearest row of ``c`` for every row of ``x`` (L2)."""
        if x.device.type == "cuda" and self.vector_len == 4:
            from ..kernels import kmeans_assign as _kmeans_assign

            return _kmeans_assign(x.contiguous(), c.contiguous(), self.vector_len)
        if x.device.type == "cuda":
            free = torch.cuda.mem_get_info(x.device)[0]
            chunk = max(1, int(free * 0.3) // (c.shape[0] * 4))
        else:
            chunk = max(1, (1 << 26) // c.shape[0])
        return torch.cat(
            [
                torch.cdist(x[s : s + chunk], c).argmin(dim=1)
                for s in range(0, x.shape[0], chunk)
            ]
        )

    def cholesky_inv_upper(self, H: torch.Tensor) -> torch.Tensor:
        """Return upper Cholesky factor U of H^{-1} (U^T U = H^{-1}).

        Applies progressive dampening on failure, matching the approach in
        :meth:`~neuropress.methods.gptq.GPTQCalibration._apply_damp_and_compute_inv_hessian_chol_upper`.

        Raises:
            RuntimeError: If Cholesky fails even at full dampening.
        """
        damp_percent = self.gptq_damp_percent
        H = H.clone().float()
        while damp_percent < 1.0:
            damp = damp_percent * H.diagonal().mean()
            H.diagonal().add_(damp)
            try:
                L = torch.linalg.cholesky(H)
                H_inv = torch.cholesky_inverse(L)
                return torch.linalg.cholesky(H_inv, upper=True)
            except torch._C._LinAlgError:  # type: ignore[attr-defined]
                LOGGER.warning(
                    "VQGPTQCorrector: Cholesky failed (damp=%.5f), retrying with +%.5f",
                    damp_percent,
                    self.gptq_damp_auto_increment,
                )
                damp_percent += self.gptq_damp_auto_increment
        raise RuntimeError(
            f"Cholesky decomposition failed (damp exhausted at {damp_percent:.4f})"
        )
