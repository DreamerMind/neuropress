"""GPTQ Q4_0 for fused MoE experts and its torch_to_nnef handoff."""

import tarfile

import numpy as np
import pytest
import torch
from torch import nn

from neuropress.adapter import CompositeAdapter, WeightAssignmentAdapter
from neuropress.calibration.probe import StatisticRequest, StatisticsRegistry
from neuropress.estimators import CompositeEstimator
from neuropress.moe_experts import (
    EXPERT_LAYOUT,
    GPTQMoEExpertsEstimator,
    MoEExpertsQ40Adapter,
    _gptq_expert,
    expert_q40_from_gptq_codes,
)
from neuropress.pipeline import QuantizationPipeline
from neuropress.recipe import QuantizationRecipe
from neuropress.recipe_config import QuantizationRecipeConfig

qwen3 = pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")

D, INTER, EXPERTS = 64, 32, 4


class _BlockWrapper(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return out[0] if isinstance(out, tuple) else out


def _block(seed: int = 0) -> _BlockWrapper:
    cfg = qwen3.Qwen3MoeConfig(
        hidden_size=D,
        moe_intermediate_size=INTER,
        num_experts=EXPERTS,
        num_experts_per_tok=2,
        hidden_act="silu",
        norm_topk_prob=True,
    )
    torch.manual_seed(seed)
    block = qwen3.Qwen3MoeSparseMoeBlock(cfg)
    with torch.no_grad():
        for p in block.parameters():
            nn.init.normal_(p, std=0.05)
    return _BlockWrapper(block).eval()


def _batches(n: int, seed: int) -> list[torch.Tensor]:
    """Anisotropic hidden states, as real residual streams are.

    Every seed draws from the same distribution, so calibration and held-out
    batches differ only in their samples.
    """
    mix_gen = torch.Generator().manual_seed(1234)
    mix = torch.randn(D, D, generator=mix_gen) * torch.logspace(0, -2, D).unsqueeze(1)
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(2, 16, D, generator=gen) @ mix for _ in range(n)]


def _recipe() -> QuantizationRecipe:
    return QuantizationRecipe(
        param_estimator=CompositeEstimator(estimators=[GPTQMoEExpertsEstimator()]),
        adapter=CompositeAdapter([WeightAssignmentAdapter(), MoEExpertsQ40Adapter()]),
    )


def test_collector_accumulates_routed_tokens_per_expert():
    model = _block()
    experts = model.block.experts
    batch = _batches(1, seed=1)[0]
    registry = StatisticsRegistry()
    request = StatisticRequest(
        target_key="block.experts", stat_type="moe_expert_hessian"
    )
    registry.register_requests([request])
    routed = []
    hook = experts.register_forward_pre_hook(lambda m, args: routed.append(args))
    registry.collect(model, [batch])
    hook.remove()
    stat = registry.get(request)

    hidden, top_k_index = routed[0][0], routed[0][1]
    for expert in range(EXPERTS):
        x = hidden[(top_k_index == expert).any(-1)]
        gate, up = nn.functional.linear(x, experts.gate_up_proj[expert]).chunk(2, -1)
        z = experts.act_fn(gate) * up
        assert stat["nsamples"][expert] == x.shape[0]
        n = max(1, x.shape[0])
        torch.testing.assert_close(stat["gate_up"][expert], 2 * x.t() @ x / n)
        torch.testing.assert_close(stat["down"][expert], 2 * z.t() @ z / n)


def test_pipeline_quantizes_experts_and_marks_the_block():
    model = _block()
    float_gate_up = model.block.experts.gate_up_proj.detach().clone()
    registry = StatisticsRegistry()
    request = StatisticRequest(
        target_key="block.experts", stat_type="moe_expert_hessian"
    )
    registry.register_requests([request])
    batches = _batches(4, seed=2)
    registry.collect(model, batches)
    stat = registry.get(request)

    QuantizationPipeline(_recipe()).run(model, batches)

    block = model.block
    assert block._t2n_quantize_moe_experts_q40 is True
    assert block._t2n_moe_expert_layout == EXPERT_LAYOUT == "linear"
    assert block._t2n_quantize_moe_experts_q40_quantizer is expert_q40_from_gptq_codes

    # Expert 1 gets the codes GPTQ picks for it alone, and the float parameter
    # holds exactly what those codes decode to.
    codes, scales = _gptq_expert(
        float_gate_up[1],
        stat["gate_up"][1],
        int(stat["nsamples"][1]),
        group_size=32,
        damp_percent=0.01,
        damp_auto_increment=0.0025,
        name="ref",
        min_tokens=GPTQMoEExpertsEstimator().min_expert_tokens,
    )
    deq = (codes.float() - 8).unflatten(-1, (-1, 32)) * scales.half().float().unsqueeze(
        -1
    )
    torch.testing.assert_close(
        block.experts.gate_up_proj[1], deq.flatten(-2), rtol=0, atol=0
    )

    # torch_to_nnef extracts w1 (gate half) in the linear layout.
    w1 = block.experts.gate_up_proj[:, :INTER].contiguous()
    q = expert_q40_from_gptq_codes(w1, **block._t2n_quantize_moe_experts_q40_kwargs)
    assert tuple(q.shape) == (EXPERTS, INTER, D)
    torch.testing.assert_close(q.decompress_to_u8()[1], codes[:INTER], rtol=0, atol=0)
    torch.testing.assert_close(q.decompress(), w1, rtol=0, atol=0)


def test_quantizer_refuses_the_canonical_layout():
    model = _block()
    QuantizationPipeline(_recipe()).run(model, _batches(2, seed=3))
    kwargs = model.block._t2n_quantize_moe_experts_q40_kwargs
    gate_up = model.block.experts.gate_up_proj
    # Canonical w1 is [E, D, I]: same shape as w2 in the linear layout, but its
    # blocks would run along the expert's output dimension.
    canonical_w1 = gate_up[:, :INTER].transpose(-1, -2).contiguous()
    with pytest.raises(ValueError, match="linear"):
        expert_q40_from_gptq_codes(canonical_w1, **kwargs)
    # A later cast of the float model still finds its codes.
    w2 = model.block.experts.down_proj.detach().to(torch.bfloat16)
    q = expert_q40_from_gptq_codes(w2, **kwargs)
    assert q.dtype == torch.bfloat16


def test_gptq_experts_beat_minmax_q40_on_held_out_tokens():
    from torch_to_nnef.tensor.quant import fp_to_tract_q4_0_with_min_max_calibration

    reference = _block()
    held_out = torch.cat(_batches(4, seed=99))
    with torch.no_grad():
        expected = reference(held_out)

    minmax = _block()
    with torch.no_grad():
        for name in ("gate_up_proj", "down_proj"):
            p = getattr(minmax.block.experts, name)
            p.copy_(fp_to_tract_q4_0_with_min_max_calibration(p.detach()).decompress())
        minmax_err = (minmax(held_out) - expected).pow(2).mean()

    gptq = _block()
    QuantizationPipeline(_recipe()).run(gptq, _batches(16, seed=4))
    with torch.no_grad():
        gptq_err = (gptq(held_out) - expected).pow(2).mean()
    assert gptq_err < 0.8 * minmax_err, (gptq_err, minmax_err)


def test_recipe_config_round_trips_through_json():
    from neuropress.t2n.registry import NEUROPRESS_COMPRESSION

    entry = NEUROPRESS_COMPRESSION["open_text_c4_gptq_q4_0_moe"]
    config = QuantizationRecipeConfig.from_json(entry.__config__.model_dump_json())
    recipe = config.build()
    kinds = {type(e).__name__ for e in recipe.param_estimator.estimators}
    assert kinds == {"GPTQEstimator", "GPTQMoEExpertsEstimator"}
    assert any(isinstance(a, MoEExpertsQ40Adapter) for a in recipe.adapter.adapters)


def _tract_target():
    from torch_to_nnef.inference_target import TractNNEF

    return TractNNEF(version="0.23.10", check_io=True)


def test_nnef_export_stores_the_gptq_codes(tmp_path):
    torch_to_nnef = pytest.importorskip("torch_to_nnef")
    from torch_to_nnef.nnef_io.tensor import DatBinHeader

    model = _block()
    QuantizationPipeline(_recipe()).run(model, _batches(4, seed=5))
    expected = {
        "w1": model.block.experts.gate_up_proj[:, :INTER],
        "w3": model.block.experts.gate_up_proj[:, INTER:],
        "w2": model.block.experts.down_proj,
    }
    path = tmp_path / "moe.nnef.tgz"
    torch_to_nnef.export_model_to_nnef(
        model=model,
        args=(torch.cat(_batches(1, seed=6))[:1],),
        file_path_export=path,
        inference_target=_tract_target(),
        input_names=["tokens"],
        output_names=["output"],
    )
    with tarfile.open(path) as archive:
        graph = next(m for m in archive.getmembers() if m.name.endswith("graph.nnef"))
        assert "expert_layout = 'linear'" in archive.extractfile(graph).read().decode()
        found = set()
        for member in archive.getmembers():
            name = next(
                (k for k in expected if member.name.endswith(f"_{k}.dat")), None
            )
            if name is None:
                continue
            archive.extract(member, tmp_path / "x", filter="data")
            with (tmp_path / "x" / member.name).open("rb") as stream:
                header = DatBinHeader.from_dat_file(stream)
                payload = np.frombuffer(stream.read(), dtype=np.uint8)
            assert header.dims == list(expected[name].shape)
            payload = payload.reshape(-1, 18)
            scales = torch.from_numpy(payload[:, :2].copy().view("<f2")).float()
            nibbles = torch.from_numpy(payload[:, 2:].astype(np.int64))
            codes = torch.cat([nibbles & 15, nibbles >> 4], dim=1).float() - 8
            torch.testing.assert_close(
                (codes * scales).reshape(expected[name].shape),
                expected[name].detach().float(),
                rtol=0,
                atol=0,
            )
            found.add(name)
    assert found == set(expected)


def test_registry_entry_quantizes_linears_and_experts_of_a_causal_lm():
    from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

    from neuropress.recipe import CalibrationConfig
    from neuropress.t2n.registry import (
        NEUROPRESS_COMPRESSION,
        run_quantization_pipeline,
    )

    from .test_t2n_register import _mock_load_dataset, _MockTokenizer

    cfg = qwen3.Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=D,
        intermediate_size=64,
        moe_intermediate_size=INTER,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        num_experts=EXPERTS,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
    )
    torch.manual_seed(0)
    model = qwen3.Qwen3MoeForCausalLM(cfg).eval()
    config = NEUROPRESS_COMPRESSION["open_text_c4_gptq_q4_0_moe"].__config__
    recipe = config.model_copy(
        update={
            "calibration": CalibrationConfig(n_samples=4, seq_len=32, seed=0),
            "kernel_lowering": None,
        }
    ).build()
    run_quantization_pipeline(
        model,
        tokenizer=_MockTokenizer(),
        recipe=recipe,
        _load_dataset=_mock_load_dataset,
    )
    for layer in model.model.layers:
        assert isinstance(layer.self_attn.q_proj.weight, QTensorTractScaleOnly)
        # the router stays float
        assert not isinstance(layer.mlp.gate.weight, QTensorTractScaleOnly)
        moe = layer.mlp
        assert moe._t2n_moe_expert_layout == "linear"
        assert moe._t2n_quantize_moe_experts_q40_quantizer is expert_q40_from_gptq_codes
        w2 = moe.experts.down_proj.detach().contiguous()
        q = expert_q40_from_gptq_codes(w2, **moe._t2n_quantize_moe_experts_q40_kwargs)
        torch.testing.assert_close(q.decompress(), w2, rtol=0, atol=0)


def test_block_without_an_extractor_is_not_marked(caplog):
    class UnknownMoeBlock(qwen3.Qwen3MoeSparseMoeBlock):
        pass

    model = _block()
    model.block.__class__ = UnknownMoeBlock
    with caplog.at_level("WARNING", logger="neuropress.moe_experts"):
        QuantizationPipeline(_recipe()).run(model, _batches(2, seed=7))
    assert not hasattr(model.block, "_t2n_quantize_moe_experts_q40")
    assert not hasattr(model.block, "_t2n_quantize_moe_experts_q40_kwargs")
    assert "no MoE extractor for UnknownMoeBlock" in caplog.text


def test_experts_keep_scales_only():
    model = _block()
    QuantizationPipeline(_recipe()).run(model, _batches(2, seed=8))
    sources = model.block._t2n_quantize_moe_experts_q40_kwargs["sources"]
    assert not any(hasattr(s, "codes") for s in sources)
    assert [s.scales.dtype for s in sources] == [torch.float16] * 3
    assert tuple(sources[0].scales.shape) == (EXPERTS, INTER, D // 32)


def test_staged_run_accepts_expert_assignments():
    model = _block()
    out = QuantizationPipeline(_recipe()).run(
        model, _batches(2, seed=9), target_paths={"block.experts"}
    )
    assert out.block._t2n_quantize_moe_experts_q40 is True


def test_checkpoint_round_trip_restores_the_export_markers(tmp_path):
    from neuropress.t2n.model_io import load_quantized_model, save_quantized_model

    model = _block()
    QuantizationPipeline(_recipe()).run(model, _batches(2, seed=10))
    path = tmp_path / "moe.safetensors"
    save_quantized_model(model, path)

    fresh = _block(seed=1)
    load_quantized_model(fresh, path, kernel_lowering=None)
    assert fresh.block._t2n_quantize_moe_experts_q40 is True
    assert fresh.block._t2n_moe_expert_layout == "linear"
    w2 = fresh.block.experts.down_proj
    before = expert_q40_from_gptq_codes(
        model.block.experts.down_proj,
        **model.block._t2n_quantize_moe_experts_q40_kwargs,
    )
    after = expert_q40_from_gptq_codes(
        w2, **fresh.block._t2n_quantize_moe_experts_q40_kwargs
    )
    torch.testing.assert_close(
        after.decompress_to_u8(), before.decompress_to_u8(), rtol=0, atol=0
    )
    torch.testing.assert_close(after.decompress(), w2, rtol=0, atol=0)
    # saving the reloaded model keeps the entry once
    save_quantized_model(fresh, tmp_path / "again.safetensors")
    again = _block(seed=2)
    manifest = load_quantized_model(again, tmp_path / "again.safetensors", None)
    entries = [
        t
        for t in manifest.transforms
        if isinstance(t, dict) and t["type"] == "moe_experts_q40"
    ]
    assert len(entries) == 1 and again.block._t2n_quantize_moe_experts_q40 is True


def test_collector_ignores_masked_slots():
    from neuropress.calibration.probe.moe_experts import MoEExpertHessianCollector

    experts = _block().block.experts
    hidden = torch.randn(5, D)
    index = torch.tensor([[0, 1], [1, EXPERTS], [2, 0], [EXPERTS, EXPERTS], [3, 1]])
    collector = MoEExpertHessianCollector()
    collector.update_call(experts, (hidden, index, torch.ones(5, 2)), None)
    stat = collector.result()
    assert stat["nsamples"].tolist() == [2, 3, 1, 1]
    x = hidden[[0, 1, 4]]
    torch.testing.assert_close(stat["gate_up"][1], 2 * x.t() @ x / 3)


def test_collector_result_is_scaled_once_and_closes_the_collector():
    from neuropress.calibration.probe.moe_experts import MoEExpertHessianCollector

    experts = _block().block.experts
    hidden = torch.randn(4, D)
    index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    collector = MoEExpertHessianCollector()
    collector.update_call(experts, (hidden, index, torch.ones(4, 2)), None)
    first = collector.result()["gate_up"].clone()
    torch.testing.assert_close(collector.result()["gate_up"], first)
    with pytest.raises(RuntimeError, match="already read"):
        collector.update_call(experts, (hidden, index, torch.ones(4, 2)), None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_collector_host_fallback_matches_the_device_hessians(monkeypatch):
    from neuropress.calibration.probe.moe_experts import MoEExpertHessianCollector

    experts = _block().block.experts.cuda()
    hidden = torch.randn(64, D, device="cuda")
    index = torch.randint(0, EXPERTS, (64, 2), device="cuda")
    results = []
    for host in (False, True):
        if host:
            monkeypatch.setattr(
                MoEExpertHessianCollector,
                "_device",
                staticmethod(lambda n, widths, device: torch.device("cpu")),
            )
        collector = MoEExpertHessianCollector()
        for _ in range(3):
            collector.update_call(experts, (hidden, index, torch.ones(64, 2)), None)
        results.append(collector.result())
    assert results[1]["gate_up"].device.type == "cpu"
    assert results[1]["down"].device.type == "cpu"
    for key in ("gate_up", "down"):
        torch.testing.assert_close(results[1][key], results[0][key].cpu())


def test_codes_for_accepts_zero_blocks_and_f16_subnormal_grids():
    from neuropress.moe_experts import _ExpertScales

    scales = torch.tensor([[[2.0**-20, 0.0]]]).half()  # subnormal, stored as zero
    levels = torch.tensor([*range(-8, 8)] * 2, dtype=torch.float32)
    values = torch.cat([levels * 2.0**-20, torch.zeros(32)])[None, None]
    codes = _ExpertScales("w", scales).codes_for(values.half())
    assert codes is not None
    assert codes[0, 0, :32].tolist() == [int(v) + 8 for v in levels]
    assert codes[0, 0, 32:].tolist() == [8] * 32


def test_experts_with_few_tokens_lean_toward_round_to_nearest():
    torch.manual_seed(0)
    weight = torch.randn(16, 64)
    x = torch.randn(2, 64)
    hessian = 2 * x.t() @ x / 2
    kwargs = {"group_size": 32, "damp_percent": 0.01, "damp_auto_increment": 0.0025}
    rtn, _ = _gptq_expert(weight, hessian, 0, name="rtn", **kwargs)
    raw, _ = _gptq_expert(weight, hessian, 2, name="raw", **kwargs)
    shrunk, _ = _gptq_expert(weight, hessian, 2, name="s", min_tokens=1000, **kwargs)
    agree = lambda a: (a == rtn).float().mean().item()  # noqa: E731
    assert agree(shrunk) > agree(raw)
    assert agree(shrunk) > 0.95


def test_few_token_expert_keeps_input_columns_its_tokens_left_at_zero():
    # Two tokens that are exactly zero on channel 5: a zero Hessian diagonal
    # entry, which GPTQ would treat as a dead input and zero the column for.
    torch.manual_seed(0)
    weight = torch.randn(16, 64)
    x = torch.randn(2, 64)
    x[:, 5] = 0
    hessian = x.t() @ x
    kwargs = {"group_size": 32, "damp_percent": 0.01, "damp_auto_increment": 0.0025}
    codes, _ = _gptq_expert(weight, hessian, 2, name="e", min_tokens=64, **kwargs)
    assert (codes[:, 5] != 8).any()


def _collected(model: nn.Module, batches: list[torch.Tensor], path: str):
    registry = StatisticsRegistry()
    request = StatisticRequest(target_key=path, stat_type="moe_expert_hessian")
    registry.register_requests([request])
    registry.collect(model, batches)
    estimator = GPTQMoEExpertsEstimator()
    estimator.prepare(model)
    return registry, request, estimator


def test_estimate_reads_the_hessians_where_they_were_collected():
    model = _block()
    registry, request, estimator = _collected(
        model, _batches(2, seed=10), "block.experts"
    )
    before = {k: v.clone() for k, v in registry.get(request).items()}
    # A relocating read would hand GPTQ meta tensors, which cannot be copied
    # back to the weights' device.
    registry.to("meta")
    plan = estimator.estimate(None, registry, _recipe())

    assert set(plan.weight_assignments) == {
        "block.experts/gate_up_proj",
        "block.experts/down_proj",
    }
    # GPTQ damps a copy, never the registry's statistic.
    stat = registry.get(request, relocate=False)
    for key, value in before.items():
        torch.testing.assert_close(stat[key], value, rtol=0, atol=0)


def test_experts_whose_scales_overflow_f16_stay_float(caplog):
    model = _block()
    with torch.no_grad():
        model.block.experts.down_proj[1, 0, :32] = 1e6
    float_params = {
        n: p.detach().clone() for n, p in model.block.experts.named_parameters()
    }
    with caplog.at_level("WARNING", logger="neuropress.moe_experts"):
        QuantizationPipeline(_recipe()).run(model, _batches(2, seed=11))

    assert "experts kept float" in caplog.text
    assert not hasattr(model.block, "_t2n_quantize_moe_experts_q40")
    for n, p in model.block.experts.named_parameters():
        torch.testing.assert_close(p, float_params[n], rtol=0, atol=0)


def test_experts_at_the_top_level_mark_the_root_block():
    model = _block().block
    QuantizationPipeline(_recipe()).run(model, _batches(2, seed=12))
    assert model._t2n_quantize_moe_experts_q40 is True
    assert model._t2n_moe_expert_layout == EXPERT_LAYOUT
