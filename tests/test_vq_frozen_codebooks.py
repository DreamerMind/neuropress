"""VQ with frozen codebooks: one GPTQ-corrected assignment, optional teacher fit."""

from copy import deepcopy

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import neuropress.t2n.qtensors  # noqa: F401  (breaks a circular import)
from neuropress.block_pipeline import BlockQuantizationPipeline
from neuropress.calibration.probe.reconstruction import (
    LinearReconstructionCollector,
    SinkProtectedReconstructionCollector,
)
from neuropress.estimators import VPTQGroupEstimator
from neuropress.pipeline import QuantizationPipeline
from neuropress.recipe_config import QuantizationRecipeConfig
from neuropress.transforms.vq import QTensorTractVQ

VL = 4


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = nn.Linear(32, 48, bias=False)
        self.down_proj = nn.Linear(48, 32, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.up_proj(x)))


def _books(tmp_path, nc=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    books = {
        key: (torch.randn(nc, VL, generator=g) * 0.12).to(torch.float16)
        for key in ("up_proj", "down_proj")
    }
    path = tmp_path / "books.safetensors"
    save_file(books, str(path))
    return books, str(path)


def _recipe(path, teacher=False, **extra):
    return QuantizationRecipeConfig.model_validate(
        {
            "estimator": {
                "type": "vptq",
                "target": "linear",
                "vector_len": VL,
                "num_centroids": 256,
                "grouping": "by_leaf_name",
                "gptq_correction_iters": 1,
                "gptq_block_ldlq": True,
                "frozen_centroids_path": path,
                "reconstruction": "teacher" if teacher else "self",
                **extra,
            },
            "kernel_lowering": None,
        }
    )


def _block(seed=0):
    torch.manual_seed(seed)
    block = _Block().eval()
    for lin in (block.up_proj, block.down_proj):
        lin.weight.data.mul_(0.6)
    return block


def test_frozen_codebooks_are_used_unchanged(tmp_path):
    books, path = _books(tmp_path)
    block = _block()
    x = [torch.randn(4, 16, 32) for _ in range(2)]
    pipeline = QuantizationPipeline(_recipe(path).build())
    pipeline.run(block, x, estimation_device="cpu")
    assert not pipeline.last_plan.kept_float
    for key in ("up_proj", "down_proj"):
        qt = pipeline.last_plan.weight_assignments[key]
        assert isinstance(qt, QTensorTractVQ)
        torch.testing.assert_close(
            qt.qscheme.centroids.cpu(), books[key], rtol=0, atol=0
        )
        # every dequantized vector is a codeword
        w = qt.decompress().float()
        perm = qt.qscheme.column_permutation
        if perm is not None:
            w = w[:, perm]
        vecs = w.reshape(-1, VL)
        dist = torch.cdist(vecs, books[key].float()).min(1).values
        assert dist.max() < 1e-3


def test_missing_group_codebook_fails(tmp_path):
    path = tmp_path / "books.safetensors"
    save_file({"up_proj": torch.zeros(16, VL, dtype=torch.float16)}, str(path))
    with pytest.raises(KeyError, match="down_proj"):
        QuantizationPipeline(_recipe(str(path)).build()).run(
            _block(), [torch.randn(2, 8, 32)], estimation_device="cpu"
        )


def test_config_validation():
    with pytest.raises(ValueError, match="frozen_centroids_path"):
        VPTQGroupEstimator(reconstruction="teacher", gptq_correction_iters=1)
    with pytest.raises(ValueError, match="gptq_correction_iters=1"):
        VPTQGroupEstimator(frozen_centroids_path="x", gptq_correction_iters=2)


def test_teacher_matches_self_on_identical_streams(tmp_path):
    _, path = _books(tmp_path)
    x = [torch.randn(4, 16, 32)]
    original = _block()
    models = [deepcopy(original), deepcopy(original)]
    for model, teacher in zip(models, [False, True], strict=True):
        QuantizationPipeline(_recipe(path, teacher).build()).run(
            model, x, reference_data=x, estimation_device="cpu"
        )
    for key in ("up_proj", "down_proj"):
        a = getattr(models[0], key).weight.decompress()
        b = getattr(models[1], key).weight.decompress()
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_teacher_corrects_known_upstream_scale_error(tmp_path):
    # A single linear isolates the fit: the teacher sees x, the student x * 1.25.
    original = _block(3)
    x = torch.randn(8, 32, 32)
    student = x * 1.25
    lin = nn.Sequential(deepcopy(original.up_proj)).eval()
    lin_plain, lin_teacher = deepcopy(lin), deepcopy(lin)
    # Codewords cover both the source weight and its 1/1.25 rescale, so the
    # comparison measures the fit, not codebook resolution.
    w = lin[0].weight.detach()
    books = {
        "0": torch.cat(
            [w.reshape(-1, VL), (w / 1.25).reshape(-1, VL), torch.zeros(256, VL)]
        ).to(torch.float16)
    }
    single = tmp_path / "single.safetensors"
    save_file(books, str(single))
    for model, use_teacher in [(lin_plain, False), (lin_teacher, True)]:
        QuantizationPipeline(_recipe(str(single), use_teacher).build()).run(
            model, [student], reference_data=[x], estimation_device="cpu"
        )
    with torch.no_grad():
        target = lin(x)
        err_teacher = (lin_teacher(student) - target).norm()
        err_plain = (lin_plain(student) - target).norm()
    assert err_teacher < 0.6 * err_plain


@pytest.mark.parametrize("source", ["float", "sequential"])
def test_staged_block_run_with_frozen_codebooks(tmp_path, source):
    books, path = _books(tmp_path)
    blocks = [_block(1), _block(2)]
    staged = source == "sequential"  # dependency stages need sequential inputs
    runner = BlockQuantizationPipeline(
        _recipe(path, teacher=staged),
        input_source=source,
        target_paths=("up_proj", "down_proj") if staged else None,
    )
    results = list(
        runner.run(
            [(f"b{i}", b) for i, b in enumerate(blocks)], [torch.randn(2, 8, 32)]
        )
    )
    if staged:
        assert [r.schedule.stages for r in results] == [
            (("up_proj",), ("down_proj",))
        ] * 2
    for block in blocks:
        for key in ("up_proj", "down_proj"):
            qt = getattr(block, key).weight
            assert isinstance(qt, QTensorTractVQ)
            torch.testing.assert_close(
                qt.qscheme.centroids.cpu(), books[key], rtol=0, atol=0
            )


def test_output_ratio_guard_drops_large_teacher_fits(tmp_path, caplog):
    # Same setup as the scale-error test: the fit moves the output by ~20%,
    # so a 5% limit falls back to the plain weight and matches self exactly.
    original = _block(3)
    x = torch.randn(8, 32, 32)
    student = x * 1.25
    lin = nn.Sequential(deepcopy(original.up_proj)).eval()
    w = lin[0].weight.detach()
    books = {
        "0": torch.cat(
            [w.reshape(-1, VL), (w / 1.25).reshape(-1, VL), torch.zeros(256, VL)]
        ).to(torch.float16)
    }
    single = tmp_path / "single.safetensors"
    save_file(books, str(single))
    plain, guarded, loose = deepcopy(lin), deepcopy(lin), deepcopy(lin)
    runs = [
        (plain, {}, False),
        (guarded, {"reconstruction_max_output_ratio": 0.05}, True),
        (loose, {"reconstruction_max_output_ratio": 10.0}, True),
    ]
    for model, extra, use_teacher in runs:
        QuantizationPipeline(_recipe(str(single), use_teacher, **extra).build()).run(
            model, [student], reference_data=[x], estimation_device="cpu"
        )
    a, b, c = (m[0].weight.decompress() for m in (plain, guarded, loose))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(a, c)
    assert "moves the output by" in caplog.text


def test_sink_protection_needs_teacher():
    with pytest.raises(ValueError, match="reconstruction_protect_sink"):
        VPTQGroupEstimator(
            frozen_centroids_path="x",
            gptq_correction_iters=1,
            reconstruction_protect_sink=True,
        )


def test_sink_collector_drops_position_zero_residual():
    lin = nn.Linear(8, 6, bias=False)
    x = torch.randn(3, 5, 8)
    out = lin(x).detach()
    ref = out + torch.randn_like(out)
    plain = LinearReconstructionCollector(module=lin)
    sink = SinkProtectedReconstructionCollector(module=lin)
    tail = LinearReconstructionCollector(module=lin)
    plain.update_pair((x,), out, ref)
    sink.update_pair((x,), out, ref)
    # the protected product equals one over positions 1.. alone, x included
    ref_tail = ref.clone()
    ref_tail[:, 0] = out[:, 0]
    tail.update_pair((x,), out, ref_tail)
    torch.testing.assert_close(sink.result()["cross"], tail.result()["cross"])
    assert not torch.allclose(sink.result()["cross"], plain.result()["cross"])
    with pytest.raises(ValueError, match="batch, seq"):
        sink.update_pair((x[0],), out[0], ref[0])


def test_sink_protected_teacher_ignores_position_zero_errors(tmp_path):
    # The student sees the teacher input except at position 0: the plain
    # teacher fit chases that token, the protected one has nothing to fit and
    # matches self exactly.
    single = tmp_path / "single.safetensors"
    save_file({"0": _books(tmp_path)[0]["up_proj"]}, str(single))
    x = torch.randn(4, 16, 32)
    student = x.clone()
    student[:, 0] *= 8.0
    original = nn.Sequential(deepcopy(_block(5).up_proj)).eval()
    models = [deepcopy(original) for _ in range(3)]
    runs = [(False, {}), (True, {}), (True, {"reconstruction_protect_sink": True})]
    for model, (use_teacher, extra) in zip(models, runs, strict=True):
        QuantizationPipeline(_recipe(str(single), use_teacher, **extra).build()).run(
            model, [student], reference_data=[x], estimation_device="cpu"
        )
    plain, teacher, protected = (m[0].weight.decompress() for m in models)
    torch.testing.assert_close(plain, protected, rtol=0, atol=0)
    assert not torch.equal(plain, teacher)
