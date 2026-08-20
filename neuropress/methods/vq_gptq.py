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
    """

    vector_len: int
    gptq_correction_alpha: float = 1.0
    gptq_damp_percent: float = 0.01
    gptq_damp_auto_increment: float = 0.0025
    centroid_attraction: float = 0.0
    centroid_attraction_window: int = 4

    def correct(
        self,
        name: str,
        mod: nn.Module,
        centroids: torch.Tensor,
        full_h: torch.Tensor,
        device: torch.device,
        column_perm: torch.Tensor | None = None,
        diag_h: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Run GPTQ-corrected VQ assignment for one layer.

        Args:
            name: Layer name (used for logging).
            mod: The module whose ``weight`` will be reassigned.
            centroids: Shared codebook ``[NC, VL]``.
            full_h: Full input Hessian ``[K, K]`` for this layer.
            device: Target CUDA device.
            column_perm: Optional Hessian-sorted column permutation ``[K]``.
            diag_h: Optional Hessian diagonal ``[K]`` for importance-weighted
                assignment within each group.  Falls back to uniform if None.

        Returns:
            Packed uint8 index tensor (same format as
            :func:`~neuropress.transforms.vq_packing.IndexPacking.pack`),
            or ``None`` when correction is infeasible (VRAM too tight,
            Cholesky failure, etc.).  The caller should keep the existing
            plain assignment when ``None`` is returned.
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

        if device.type != "cuda":
            return None
        w_fp32_bytes = w.numel() * 4
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if w_fp32_bytes > free_bytes * 0.4:
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

        dead = H.diagonal() == 0
        H.diagonal()[dead] = 1.0
        w2d[:, dead] = 0.0

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

        if diag_h is not None:
            diag_h = diag_h.float().to(device)
            if column_perm is not None:
                diag_h = diag_h[column_perm.to(device)]

        C = centroids.float().to(device)
        NC = C.shape[0]
        G = K // VL
        U_diag = U.diagonal()

        w_corr = w2d.clone()
        all_indices: list[torch.Tensor] = []

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

            if VL == 4:
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
        return IndexPacking.pack(indices, bits)

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
