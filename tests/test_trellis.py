"""Trellis codes and the trellis weight quantizer: torch parts on CPU, the Viterbi kernel on CUDA."""

import importlib.util
import itertools

import pytest
import torch
from torch import nn

from neuropress.methods.block_ldlq import block_columns, block_order
from neuropress.methods.trellis import (
    TILE_PERM,
    TrellisConfig,
    TrellisWeight,
    _from_walks,
    _rotate,
    _to_walks,
    _unrotate,
)
from neuropress.methods.trellis_code import TrellisCode, decode_reference, rate_pattern
from neuropress.trellis_estimator import TrellisEstimator, TrellisEstimatorConfig

# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------


def test_rate_pattern():
    assert rate_pattern(3) == (3, 0)
    assert rate_pattern(3.5) == (3, 0xAAAA)  # EXL3's half step
    ka, mask = rate_pattern(2.25)
    assert ka == 2 and bin(mask).count("1") == 4
    assert rate_pattern(8) == (8, 0)
    for bad in (0.5, 3.3, 8.5):
        with pytest.raises(ValueError, match="1/16"):
            rate_pattern(bad)


def test_code_validation():
    TrellisCode(3.5, "mcg")
    with pytest.raises(ValueError, match="multiple of 32"):
        TrellisCode(3, tile_len=48)
    with pytest.raises(ValueError, match="lut"):
        TrellisCode(3, "lut")
    with pytest.raises(ValueError, match="lut"):
        TrellisCode(3, "mul1", lut=torch.zeros(65536))
    assert TrellisCode(3.5).tile_bits == 256 * 3 + 128
    assert TrellisCode(3.25, tile_len=32).tile_words == 7  # 104 bits, padded


@pytest.mark.parametrize("bits", [1, 2.5, 3, 3.25, 3.5, 7.5, 8])
@pytest.mark.parametrize("tile_len", [32, 256])
def test_pack_round_trip(bits, tile_len):
    """Any bit stream is a closed walk: unpack then pack gives it back, and vice versa."""
    code = TrellisCode(bits, tile_len=tile_len)
    g = torch.Generator().manual_seed(0)
    words = torch.randint(-32768, 32768, (5, 3, code.tile_words), generator=g).short()
    pad = code.tile_words * 16 - code.tile_bits
    if pad:
        words[..., -1] &= ~((1 << pad) - 1)  # padding bits are zero
    states = code.unpack(words)
    assert states.shape == (5, 3, tile_len)
    torch.testing.assert_close(code.pack(states), words)
    torch.testing.assert_close(code.unpack(code.pack(states)), states)


def test_consecutive_windows_overlap():
    code = TrellisCode(3.5, tile_len=64)
    g = torch.Generator().manual_seed(1)
    s = code.unpack(
        torch.randint(-32768, 32768, (4, code.tile_words), generator=g).short()
    )
    s = s.long() & 0xFFFF
    w = code.widths
    for i in range(64):
        j = (i + 1) % 64
        keep = 16 - int(w[j])
        assert torch.equal(s[:, i] & ((1 << keep) - 1), s[:, j] >> int(w[j]))


def test_decode_reference_ranges():
    s = torch.arange(65536)
    for cb in ("mul1", "mcg", "3inst"):
        v = decode_reference(s, cb)
        assert v.isfinite().all()
        assert 0.6 < v.square().mean().sqrt() < 2.0, cb
    lut = torch.linspace(-1, 1, 65536).half()
    torch.testing.assert_close(decode_reference(s[:10], "lut", lut), lut[:10].float())


def test_rotation_is_orthogonal_and_inverts():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 256, generator=g)
    s = torch.randint(0, 2, (256,), generator=g).float() * 2 - 1
    r = _rotate(x, s)
    torch.testing.assert_close(r.norm(dim=1), x.norm(dim=1))
    torch.testing.assert_close(_unrotate(r, s), x, rtol=1e-5, atol=1e-5)


def test_walk_layout_round_trip():
    z = torch.randn(64, 16)
    w = _to_walks(z)
    assert w.shape == (4, 256)
    torch.testing.assert_close(_from_walks(w), z)
    # element [c, r] of tile a is z[16a + r, c]; walk position p holds element TILE_PERM[p]
    p = 37
    e = int(TILE_PERM[p])
    assert w[2, p] == z[32 + e % 16, e // 16]


def test_block_order_puts_heavy_blocks_first():
    d = torch.ones(64)
    d[40:48] = 100.0  # block 2 (columns 32-47) is the heaviest
    d[0:16] = 10.0
    h = torch.diag(d)
    order = block_order(h, "act", 16)
    assert order[:2].tolist() == [2, 0]
    cols = block_columns(order, 16)
    assert sorted(cols.tolist()) == list(range(64))
    assert cols[:16].tolist() == list(range(32, 48))
    assert block_order(h, "natural", 16).tolist() == [0, 1, 2, 3]
    assert block_order(h, "reverse", 16).tolist() == [3, 2, 1, 0]


def _random_weight(n=128, k=256, bits=3.0, seed=0) -> TrellisWeight:
    g = torch.Generator().manual_seed(seed)
    code = TrellisCode(bits)
    return TrellisWeight(
        code,
        torch.randint(
            -32768, 32768, (n // 16, k // 16, code.tile_words), generator=g
        ).short(),
        (torch.rand(n, generator=g) * 0.01 + 0.001).half(),
        (torch.rand(k, generator=g) + 0.5).half(),
        (torch.randint(0, 2, (n,), generator=g) * 2 - 1).char(),
        (torch.randint(0, 2, (k,), generator=g) * 2 - 1).char(),
    )


def test_dequantize_matches_the_definition():
    """W = B diag(row_scale) T diag(col_scale) A^T, A = diag(sign_in) H, B = diag(sign_out) H."""
    from neuropress.methods.trellis import _hadamard

    tw = _random_weight(128, 256, 3.25)
    n, k = tw.shape
    vals = tw.code.decode(tw.code.unpack(tw.packed)).reshape(n // 16, k // 16, 256)
    t = torch.empty(n, k)
    for a in range(n // 16):
        for b in range(k // 16):
            tile = torch.empty(256)
            tile[TILE_PERM] = vals[a, b]  # walk position p holds element TILE_PERM[p]
            t[16 * a : 16 * a + 16, 16 * b : 16 * b + 16] = tile.view(16, 16).T
    h = _hadamard(torch.device("cpu"))
    hn = torch.block_diag(*[h] * (n // 128))
    hk = torch.block_diag(*[h] * (k // 128))
    big_a = torch.diag(tw.sign_in.float()) @ hk
    big_b = torch.diag(tw.sign_out.float()) @ hn
    ref = (
        big_b
        @ torch.diag(tw.row_scale.float())
        @ t
        @ torch.diag(tw.col_scale.float())
        @ big_a.T
    )
    torch.testing.assert_close(tw.dequantize(), ref, rtol=1e-4, atol=1e-6)


def test_bpw_counts_scales_and_signs():
    tw = _random_weight(128, 128, 3.0)
    assert tw.bpw == pytest.approx(3 + 17 * 256 / 128**2)


def test_qtensor_round_trip_through_serde(tmp_path):
    from neuropress.t2n.qtensors import load_qtensor, save_qtensor
    from neuropress.transforms.trellis import QTensorTrellis

    tw = _random_weight(bits=3.5)
    q = QTensorTrellis.from_weight(tw, torch.float32)
    dense = tw.dequantize()
    assert dense.shape == (128, 256)
    torch.testing.assert_close(q.decompress(), dense)
    save_qtensor(q, tmp_path / "w.safetensors")
    q2 = load_qtensor(tmp_path / "w.safetensors")
    assert isinstance(q2, QTensorTrellis)
    assert q2.qscheme.code == tw.code
    torch.testing.assert_close(q2.decompress(), dense)


def test_qtensor_moves_without_requantizing():
    from neuropress.transforms.trellis import QTensorTrellis

    q = QTensorTrellis.from_weight(_random_weight(), torch.float32)
    assert q.to("cpu") is q
    qb = q.to(torch.bfloat16)
    assert isinstance(qb, QTensorTrellis) and qb.u8_blob is q.u8_blob
    assert qb.decompress().dtype == torch.bfloat16
    lin = nn.Linear(256, 128, bias=False)
    lin._parameters["weight"] = q
    lin.to(torch.float64)
    assert isinstance(lin.weight, QTensorTrellis)


def test_qtensor_in_a_linear():
    from neuropress.transforms.trellis import QTensorTrellis

    tw = _random_weight()
    lin = nn.Linear(256, 128, bias=False)
    lin._parameters["weight"] = QTensorTrellis.from_weight(tw, torch.float32)
    x = torch.randn(3, 256)
    torch.testing.assert_close(lin(x), x @ tw.dequantize().T, rtol=1e-4, atol=1e-5)


def test_bits_lookup_full_name_then_leaf():
    est = TrellisEstimator(
        config=TrellisConfig(bits=3),
        bits_per_group={"q_proj": 3.5, "model.layers.15.self_attn.q_proj": 3.0},
    )
    assert est.bits_for("model.layers.3.self_attn.q_proj") == 3.5
    assert est.bits_for("model.layers.15.self_attn.q_proj") == 3.0
    assert est.bits_for("model.layers.3.mlp.up_proj") == 3
    with pytest.raises(ValueError, match="bits_per_group"):
        TrellisEstimator(bits_per_group={"q_proj": 3.3})


def _two_blocks() -> nn.Module:
    return nn.Sequential(
        *[
            nn.ModuleDict({"a": nn.Linear(128, 128), "b": nn.Linear(128, 128)})
            for _ in range(2)
        ]
    )


def test_seed_offsets_survive_target_selection():
    """A staged run seeds each linear as the whole-model run does."""
    model = _two_blocks()
    est = TrellisEstimator()
    est.prepare(model)
    whole = dict(est._seed_offsets)
    assert whole == {"0.a": 0, "0.b": 1, "1.a": 2, "1.b": 3}
    with est.select_targets({"1.b"}):
        est.prepare(model)
        assert list(est._target_modules) == ["1.b"]
        assert est._seed_offsets == whole


def test_seed_offsets_and_bits_of_a_block_are_the_whole_models():
    """A block held at its model paths keys bits and seeds as the whole model does."""
    from neuropress.block_pipeline import BlockAtPaths

    model = _two_blocks()
    est = TrellisEstimator(bits_per_group={"1.b": 2.0})
    est.prepare(model)
    whole = dict(est._seed_offsets)
    est.prepare(BlockAtPaths(model, ["1"], entry="1"))
    assert list(est._target_modules) == ["1.a", "1.b"]
    assert est._seed_offsets == whole
    assert est.bits_for("1.b") == 2.0


def test_block_seed_offsets_ignore_modules_earlier_blocks_replaced():
    from neuropress.block_pipeline import BlockAtPaths
    from neuropress.estimators import model_layout

    model = _two_blocks()
    est = TrellisEstimator()
    est.prepare(model)
    whole = dict(est._seed_offsets)
    layout = model_layout(model)
    # An earlier block's adapter split "0.a" into two linears.
    model[0]["a"] = nn.Sequential(nn.Linear(128, 64), nn.Linear(128, 64))
    est.prepare(BlockAtPaths(model, ["1"], entry="1", layout=layout))
    assert est._seed_offsets == whole
    est.prepare(BlockAtPaths(model, ["1"], entry="1"))
    assert est._seed_offsets["1.a"] == whole["1.a"] + 1


def test_scheme_moves_the_lut():
    from neuropress.transforms.trellis import TrellisScheme

    tw = _random_weight()
    code = TrellisCode(3, "lut", 256, torch.randn(65536).half())
    scheme = TrellisScheme(code, tw.row_scale, tw.col_scale, tw.sign_out, tw.sign_in)
    scheme.to_device("meta")
    assert scheme.code.lut.device.type == "meta"
    assert scheme.row_scale.device.type == "meta"


def test_config_registered_and_builds():
    from neuropress.recipe_config import QuantizationRecipeConfig

    recipe = QuantizationRecipeConfig.model_validate(
        {
            "estimator": {
                "type": "trellis",
                "bits": 4,
                "bits_per_group": {"o_proj": 3.25},
            },
            "kernel_lowering": None,
        }
    )
    est = recipe.estimator.build()
    assert isinstance(est, TrellisEstimator)
    assert est.bits_for("model.layers.0.self_attn.o_proj") == 3.25
    assert est.filter_fn("lm_head", nn.Linear(1, 1)) is False
    assert est.store == "packed"
    assert TrellisEstimatorConfig().order == "act"


# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
_HAS_EXL3 = (
    torch.cuda.is_available() and importlib.util.find_spec("exllamav3") is not None
)


def _gaussian_tiles(n, tile_len=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, tile_len, generator=g) * 1.2).cuda()


@needs_cuda
@pytest.mark.parametrize("cb", ["mul1", "mcg", "3inst"])
def test_kernel_decode_matches_reference(cb):
    s = torch.arange(65536, dtype=torch.int32).short().cuda()
    code = TrellisCode(3, cb)
    torch.testing.assert_close(
        code.decode(s).cpu(), decode_reference(s.cpu(), cb), rtol=0, atol=0
    )


@needs_cuda
@pytest.mark.parametrize("bits", [1, 2, 3, 3.25, 3.5, 4, 8])
def test_encoded_walks_are_closed_and_decode(bits):
    code = TrellisCode(bits)
    q, s = code.encode(_gaussian_tiles(6))
    torch.testing.assert_close(code.unpack(code.pack(s)), s)
    torch.testing.assert_close(code.decode(s), q, rtol=0, atol=0)


@needs_cuda
def test_error_falls_with_rate():
    x = _gaussian_tiles(32)
    errs = [
        ((TrellisCode(b).encode(x)[0] - x) ** 2).mean().item()
        for b in (1, 2, 3, 3.25, 3.5, 4)
    ]
    assert all(a > b for a, b in itertools.pairwise(errs)), errs
    # about 6 dB per bit on Gaussian data; ask for 4.5
    assert errs[2] / errs[5] > 10**0.45


@needs_cuda
def test_lut_codebook_reproduces_the_hash_codebook():
    table = decode_reference(torch.arange(65536), "mul1").half()
    x = _gaussian_tiles(8, tile_len=64)
    _, s_hash = TrellisCode(3, "mul1", 64).encode(x)
    _, s_lut = TrellisCode(3, "lut", 64, table).encode(x)
    torch.testing.assert_close(s_lut, s_hash)


@pytest.mark.skipif(not _HAS_EXL3, reason="needs CUDA and exllamav3")
@pytest.mark.parametrize(
    ("bits", "cb"), [(2, "mul1"), (3, "mul1"), (3, "mcg"), (4, "3inst"), (3.5, "mul1")]
)
def test_identical_to_exllamav3(bits, cb):
    from exllamav3.modules.quant.exl3_lib import quantize as xq

    x = _gaussian_tiles(16)
    args = {"K": bits if bits % 1 else int(bits), "devices": [0]}
    if cb != "3inst":
        args[cb] = True
    q_ref, s_ref = xq.quantize_tiles(x, args)
    q, s = TrellisCode(bits, cb).encode(x)
    torch.testing.assert_close(s, s_ref, rtol=0, atol=0)
    torch.testing.assert_close(q, q_ref, rtol=0, atol=0)


def _layer(out=256, inp=256, seed=0, massive=None):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out, inp, generator=g) * 0.02
    x = torch.randn(4096, inp, generator=g) * torch.linspace(0.2, 2.0, inp)
    if massive is not None:
        x[:, massive] *= 100.0
    h = x.T @ x / x.shape[0]
    return w.cuda(), h.cuda()


def _rel(w, wq, h):
    e = w - wq
    return (torch.trace(e @ h @ e.T) / torch.trace(w @ h @ w.T)).item()


@needs_cuda
def test_quantized_error_falls_with_bits():
    from neuropress.methods.trellis import trellis_quantize

    w, h = _layer()
    errs = [
        _rel(w, trellis_quantize(w, h, TrellisConfig(bits=b)).dequantize(), h)
        for b in (2, 3, 3.5, 4)
    ]
    assert all(a > b for a, b in itertools.pairwise(errs)), errs
    assert errs[1] / errs[3] > 10**0.45


@needs_cuda
def test_act_order_survives_a_massive_channel():
    from neuropress.methods.trellis import trellis_quantize

    w, h = _layer(massive=37)
    act = _rel(
        w, trellis_quantize(w, h, TrellisConfig(bits=3, order="act")).dequantize(), h
    )
    nat = _rel(
        w,
        trellis_quantize(w, h, TrellisConfig(bits=3, order="natural")).dequantize(),
        h,
    )
    assert act <= nat * 1.05


@needs_cuda
def test_dequantize_on_cpu_matches_cuda():
    from neuropress.methods.trellis import trellis_quantize

    w, h = _layer(128, 256)
    tw = trellis_quantize(w, h, TrellisConfig(bits=3.5))
    torch.testing.assert_close(
        tw.to("cpu").dequantize(), tw.dequantize().cpu(), rtol=1e-4, atol=1e-6
    )


@needs_cuda
@pytest.mark.parametrize("store", ["packed", "dense"])
def test_pipeline_end_to_end(store, tmp_path):
    from neuropress.pipeline import QuantizationPipeline
    from neuropress.recipe_config import QuantizationRecipeConfig
    from neuropress.transforms.trellis import QTensorTrellis

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up_proj = nn.Linear(128, 256, bias=False)
            self.down_proj = nn.Linear(256, 128, bias=False)

        def forward(self, x):
            return self.down_proj(torch.relu(self.up_proj(x)))

    torch.manual_seed(0)
    model = Block().eval()
    ref = Block().eval()
    ref.load_state_dict(model.state_dict())
    x = [torch.randn(8, 32, 128) for _ in range(2)]
    recipe = QuantizationRecipeConfig.model_validate(
        {
            "estimator": {
                "type": "trellis",
                "bits": 4,
                "bits_per_group": {"down_proj": 3.25},
                "store": store,
            },
            "kernel_lowering": None,
        }
    )
    pipeline = QuantizationPipeline(recipe.build())
    pipeline.run(model, x, estimation_device="cuda")
    assert set(pipeline.last_plan.weight_assignments) == {"up_proj", "down_proj"}
    for name in ("up_proj", "down_proj"):
        wq = getattr(model, name).weight
        assert isinstance(wq, QTensorTrellis) == (store == "packed")
        dense = wq.decompress() if store == "packed" else wq
        w = getattr(ref, name).weight
        assert dense.shape == w.shape
        assert 0 < ((dense.cpu() - w).norm() / w.norm()).item() < 0.2
    # the pipeline leaves the model on the estimation device
    model.cpu()
    assert model.down_proj.weight.device.type == "cpu"
    with torch.no_grad():
        y, y_ref = model(x[0]), ref(x[0])
    assert ((y - y_ref).norm() / y_ref.norm()).item() < 0.2
    if store == "packed":
        from neuropress.t2n.model_io import load_quantized_model, save_quantized_model

        save_quantized_model(model, tmp_path / "m.safetensors")
        fresh = Block().eval()
        load_quantized_model(fresh, tmp_path / "m.safetensors", kernel_lowering=None)
        assert isinstance(fresh.down_proj.weight, QTensorTrellis)
        with torch.no_grad():
            torch.testing.assert_close(fresh(x[0]), y)


@needs_cuda
@pytest.mark.parametrize("bits", [1, 2.5, 3, 3.25, 3.5, 7.5, 8])
@pytest.mark.parametrize("tile_len", [32, 256])
def test_cuda_unpack_matches_torch(bits, tile_len):
    code = TrellisCode(bits, tile_len=tile_len)
    g = torch.Generator().manual_seed(2)
    words = torch.randint(-32768, 32768, (7, code.tile_words), generator=g).short()
    pad = code.tile_words * 16 - code.tile_bits
    if pad:
        words[..., -1] &= ~((1 << pad) - 1)
    torch.testing.assert_close(code.unpack(words.cuda()).cpu(), code.unpack(words))


@needs_cuda
def test_estimator_passes_every_config_field(monkeypatch):
    """Per-layer settings come from the estimator config, with only bits and seed changed."""
    import neuropress.trellis_estimator as te

    seen = {}

    def fake_quantize(w, h, cfg):
        seen[cfg.seed] = cfg
        n, k = w.shape
        return _random_weight(n, k, cfg.bits).to(w.device)

    monkeypatch.setattr(te, "trellis_quantize", fake_quantize)
    model = _two_blocks()
    est = TrellisEstimator(
        config=TrellisConfig(bits=3, seed=10, scale_grid=(0.8, 0.9), scale_tiles=64),
        bits_per_group={"b": 3.5},
    )
    est.prepare(model)
    monkeypatch.setattr(est, "_hessian", lambda name, registry: torch.eye(128))
    with est.select_targets({"1.a", "1.b"}):
        est.prepare(model)
        est.estimate(None, None, None)
    assert sorted(seen) == [12, 13]
    assert seen[13].bits == 3.5 and seen[12].bits == 3
    assert all(
        c.scale_grid == (0.8, 0.9) and c.scale_tiles == 64 for c in seen.values()
    )
