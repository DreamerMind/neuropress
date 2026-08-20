"""Tests for ECVQ post-convergence annealing in VPTQGroupEstimator."""

import math

import torch

from neuropress.estimator_configs import VptqGroupEstimatorConfig
from neuropress.estimators import VPTQGroupEstimator
from neuropress.transforms.vq_packing import IndexPacking

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_estimator(**kwargs) -> VPTQGroupEstimator:
    cfg = VptqGroupEstimatorConfig(
        target="linear",
        num_centroids=16,
        vector_len=2,
        grouping="all",
        use_hessian_weighting=False,
        centroid_weighting="uniform",
        max_n_iter=20,
        tol=1e-6,
        gptq_correction_iters=0,
        **kwargs,
    )
    return cfg.build()


def _packed_entropy(packed: torch.Tensor, nc: int, n_indices: int) -> float:
    bits = math.ceil(math.log2(nc))
    idx = IndexPacking.unpack(packed, bits, n_indices)
    counts = torch.bincount(idx.long(), minlength=nc).float()
    probs = counts / counts.sum()
    p_nz = probs[probs > 0]
    return -(p_nz * p_nz.log2()).sum().item()


def _run_lloyd_internal(nc: int, vl: int, w: torch.Tensor, **est_kwargs):
    """Run VQ via internal methods, return (packed_per_layer, centroids, est, members, bits).

    Mirrors the main Lloyd loop in VPTQGroupEstimator.prepare(), including the
    ECVQ in-loop bias when ecvq_beta_in_loop > 0 and iteration >= ecvq_warmup_iters.
    """
    bits = math.ceil(math.log2(nc))
    mod = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
    mod.weight = torch.nn.Parameter(w.clone())
    members = [("layer", mod)]

    est = VPTQGroupEstimator(
        module_classes=(torch.nn.Linear,),
        num_centroids=nc,
        vector_len=vl,
        grouping="all",
        use_hessian_weighting=False,
        centroid_weighting="uniform",
        max_n_iter=30,
        tol=1e-6,
        gptq_correction_iters=0,
        **est_kwargs,
    )
    est._device = torch.device("cpu")

    # Collect all weight vectors for k-means init
    all_vecs = w.float().reshape(-1, vl)
    perm = torch.randperm(all_vecs.shape[0])[:nc]
    centroids = all_vecs[perm].clone()

    # Initial assignment
    packed_per_layer = {}
    for name, m in members:
        packed = est._assign_layer(name, m, centroids, None, column_perm=None)
        if packed is not None:
            packed_per_layer[name] = packed

    # Lloyd iterations - mirrors the M+E loop in prepare() including ECVQ in-loop.
    ecvq_mean_l2 = None
    for iteration in range(30):
        shift, *_ = est._refine_centroids(
            members, packed_per_layer, centroids, {}, registry=None
        )
        if shift < 1e-6:
            break

        # ECVQ in-loop: after warmup, bias E-step with normalized log(freq).
        centroid_bias = None
        if est.ecvq_beta_in_loop > 0 and iteration >= est.ecvq_warmup_iters:
            global_freq = est._compute_global_freq(members, packed_per_layer, nc, bits)
            if ecvq_mean_l2 is None:
                ecvq_mean_l2 = est._compute_mean_global_l2(
                    members, packed_per_layer, centroids, bits
                )
            log_nc = math.log(nc)
            centroid_bias = (est.ecvq_beta_in_loop * ecvq_mean_l2 / log_nc) * (
                global_freq + 1e-10
            ).log()

        for name, m in members:
            new_packed = est._assign_layer(
                name,
                m,
                centroids,
                None,
                column_perm=None,
                centroid_bias=centroid_bias,
            )
            if new_packed is not None:
                packed_per_layer[name] = new_packed

    return packed_per_layer, centroids, est, members, bits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEcvqAnnealingConfig:
    def test_default_schedule_is_empty(self):
        est = _make_estimator()
        assert est.ecvq_beta_schedule == ()

    def test_schedule_and_iters_set_correctly(self):
        est = _make_estimator(ecvq_beta_schedule=(0.5, 1.0), ecvq_iters_per_step=3)
        assert est.ecvq_beta_schedule == (0.5, 1.0)
        assert est.ecvq_iters_per_step == 3

    def test_in_loop_defaults_zero(self):
        est = _make_estimator()
        assert est.ecvq_beta_in_loop == 0.0
        assert est.ecvq_warmup_iters == 20

    def test_in_loop_fields_set(self):
        est = _make_estimator(ecvq_beta_in_loop=0.05, ecvq_warmup_iters=10)
        assert est.ecvq_beta_in_loop == 0.05
        assert est.ecvq_warmup_iters == 10

    def test_config_vd_defaults_match_estimator(self):
        cfg = VptqGroupEstimatorConfig(target="linear")
        est = VPTQGroupEstimator(module_classes=(torch.nn.Linear,))
        assert cfg.ecvq_beta_schedule == est.ecvq_beta_schedule
        assert cfg.ecvq_iters_per_step == est.ecvq_iters_per_step
        assert cfg.ecvq_beta_in_loop == est.ecvq_beta_in_loop
        assert cfg.ecvq_warmup_iters == est.ecvq_warmup_iters


class TestEcvqAnnealingMechanism:
    def test_log_freq_bias_form(self):
        """centroid_bias uses normalized log(freq): β·(mean_l2/log(NC))·log(freq)."""
        nc = 16
        freq = torch.full((nc,), 1.0 / nc)
        beta = 1.5
        mean_l2 = 0.05
        log_nc = math.log(nc)

        computed = (beta * mean_l2 / log_nc) * (freq + 1e-10).log()

        # Must differ from linear form
        linear = beta * freq
        assert not torch.allclose(computed, linear)

        # log form is always <= 0 for freq in (0,1]; linear form always > 0
        assert (computed <= 0).all()
        assert (linear > 0).all()

        # Normalization: at uniform the absolute max bias equals beta * mean_l2
        # (since |log(1/NC)| / log(NC) = 1)
        max_abs = computed.abs().max().item()
        assert abs(max_abs - beta * mean_l2) < 1e-5

    def test_log_freq_bias_prefers_popular_centroids(self):
        """Higher-frequency centroids get a larger (less negative) bias -> cheaper assignment."""
        nc = 16
        torch.manual_seed(7)
        freq = torch.rand(nc)
        freq = freq / freq.sum()
        beta = 1.0

        bias = beta * (freq + 1e-10).log()

        most_popular = freq.argmax()
        least_popular = freq.argmin()
        assert bias[most_popular] > bias[least_popular]

    def test_empty_schedule_skips_annealing(self, monkeypatch):
        """Empty ecvq_beta_schedule: _compute_global_freq NOT called for annealing."""
        calls = []
        original = VPTQGroupEstimator._compute_global_freq

        def spy(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(VPTQGroupEstimator, "_compute_global_freq", spy)

        # Run with empty schedule
        nc, vl = 16, 4
        torch.manual_seed(0)
        w = torch.randn(16, 8)
        _run_lloyd_internal(
            nc, vl, w
        )  # uses default empty schedule via VPTQGroupEstimator defaults

        # _compute_global_freq should not be called at all
        # (it's only used in the ECVQ annealing block and concentration_beta, both off here)
        assert len(calls) == 0

    def test_annealing_changes_assignments(self):
        """Running the ECVQ annealing E-step with a log-freq bias changes assignments."""
        nc, vl = 16, 2
        torch.manual_seed(42)
        # Skewed distribution: most vectors cluster near 2 modes
        modes = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
        parts = [modes[i % 2] + torch.randn(2) * 0.05 for i in range(200)]
        w = torch.stack(parts).reshape(20, 20)

        packed_per_layer, centroids, est, members, bits = _run_lloyd_internal(nc, vl, w)
        n_vecs = w.shape[0] * (w.shape[1] // vl)

        # Baseline packed (after Lloyd convergence)
        baseline_packed = packed_per_layer["layer"].clone()

        # Apply one ECVQ E-step with large beta
        freq = est._compute_global_freq(members, packed_per_layer, nc, bits)
        bias = 5.0 * (freq + 1e-10).log()
        new_packed = est._assign_layer(
            "layer", members[0][1], centroids, None, centroid_bias=bias
        )

        # Assignments must change under strong bias
        baseline_idx = IndexPacking.unpack(baseline_packed, bits, n_vecs)
        new_idx = IndexPacking.unpack(new_packed, bits, n_vecs)
        assert not torch.equal(baseline_idx, new_idx), (
            "Expected ECVQ bias to change some assignments"
        )

    def test_annealing_lowers_entropy(self):
        """After ECVQ E+M steps, entropy of assignments is lower than baseline."""
        nc, vl = 16, 2
        torch.manual_seed(42)
        modes = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
        parts = [modes[i % 2] + torch.randn(2) * 0.05 for i in range(200)]
        w = torch.stack(parts).reshape(20, 20)

        packed_per_layer, centroids, est, members, bits = _run_lloyd_internal(nc, vl, w)
        n_vecs = w.shape[0] * (w.shape[1] // vl)

        baseline_packed = packed_per_layer["layer"].clone()
        H_before = _packed_entropy(baseline_packed, nc, n_vecs)

        # Run 5 ECVQ iterations with beta=3.0
        for _ in range(5):
            freq = est._compute_global_freq(members, packed_per_layer, nc, bits)
            bias = 3.0 * (freq + 1e-10).log()
            for name, mod in members:
                new_packed = est._assign_layer(
                    name, mod, centroids, None, centroid_bias=bias
                )
                if new_packed is not None:
                    packed_per_layer[name] = new_packed
            est._refine_centroids(
                members, packed_per_layer, centroids, {}, registry=None
            )

        H_after = _packed_entropy(packed_per_layer["layer"], nc, n_vecs)

        assert H_after < H_before, (
            f"ECVQ annealing should lower entropy: before={H_before:.3f}, after={H_after:.3f}"
        )


class TestEcvqInLoop:
    def test_in_loop_bias_applied_after_warmup(self):
        """With ecvq_beta_in_loop>0, assignments differ from beta=0 after warmup."""
        nc, vl = 16, 2
        torch.manual_seed(42)
        modes = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
        parts = [modes[i % 2] + torch.randn(2) * 0.05 for i in range(200)]
        w = torch.stack(parts).reshape(20, 20)

        # Run baseline (no in-loop ECVQ)
        packed_base, _centroids_base, _, _, bits = _run_lloyd_internal(nc, vl, w)

        # Run with in-loop ECVQ from iter 0 (warmup=0 so it's always active)
        packed_ecvq, _centroids_ecvq, _est_ecvq, _members_ecvq, _ = _run_lloyd_internal(
            nc,
            vl,
            w,
            ecvq_beta_in_loop=2.0,
            ecvq_warmup_iters=0,
        )

        n_vecs = w.shape[0] * (w.shape[1] // vl)
        idx_base = IndexPacking.unpack(packed_base["layer"], bits, n_vecs)
        idx_ecvq = IndexPacking.unpack(packed_ecvq["layer"], bits, n_vecs)
        assert not torch.equal(idx_base, idx_ecvq), (
            "In-loop ECVQ bias should change assignments vs no-bias baseline"
        )

    def test_in_loop_lowers_entropy_vs_baseline(self):
        """In-loop ECVQ (beta>0) produces lower assignment entropy than beta=0."""
        nc, vl = 16, 2
        torch.manual_seed(42)
        modes = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
        parts = [modes[i % 2] + torch.randn(2) * 0.05 for i in range(200)]
        w = torch.stack(parts).reshape(20, 20)

        n_vecs = w.shape[0] * (w.shape[1] // vl)

        packed_base, _, _, _, _bits = _run_lloyd_internal(nc, vl, w)
        H_base = _packed_entropy(packed_base["layer"], nc, n_vecs)

        packed_ecvq, _, _, _, _ = _run_lloyd_internal(
            nc,
            vl,
            w,
            ecvq_beta_in_loop=2.0,
            ecvq_warmup_iters=0,
        )
        H_ecvq = _packed_entropy(packed_ecvq["layer"], nc, n_vecs)

        assert H_ecvq < H_base, (
            f"In-loop ECVQ should lower entropy: base={H_base:.3f}, ecvq={H_ecvq:.3f}"
        )

    def test_zero_beta_no_effect(self):
        """ecvq_beta_in_loop=0 (default) produces identical results to omitting it."""
        nc, vl = 16, 2
        torch.manual_seed(0)
        w = torch.randn(16, 8)

        # Reset seed before each call so centroid initialization is identical.
        torch.manual_seed(1)
        packed_a, _, _, _, bits = _run_lloyd_internal(nc, vl, w)
        torch.manual_seed(1)
        packed_b, _, _, _, _ = _run_lloyd_internal(
            nc,
            vl,
            w,
            ecvq_beta_in_loop=0.0,
            ecvq_warmup_iters=20,
        )
        n_vecs = w.shape[0] * (w.shape[1] // vl)
        idx_a = IndexPacking.unpack(packed_a["layer"], bits, n_vecs)
        idx_b = IndexPacking.unpack(packed_b["layer"], bits, n_vecs)
        assert torch.equal(idx_a, idx_b), "beta=0 must be a no-op"
