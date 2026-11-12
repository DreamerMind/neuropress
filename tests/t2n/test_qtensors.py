"""Tests for QTensor safetensors serialization (neuropress.t2n.qtensors)."""

import pytest
import torch
from torch_to_nnef.tensor.quant import fp_to_tract_q4_0_with_min_max_calibration
from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

from neuropress.t2n.qtensors import (
    _SERDE_REGISTRY,
    QTensorSerde,
    load_qtensor,
    register_qtensor_serde,
    save_qtensor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GROUP_SIZE = 32


def _make_q4_0(rows: int = 32, cols: int = 64) -> QTensorTractScaleOnly:
    """Create a QTensorTractScaleOnly using the production path."""
    w = torch.randn(rows, cols)
    q = fp_to_tract_q4_0_with_min_max_calibration(w)
    return q


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_u8_blob(tmp_path):
    q = _make_q4_0()
    path = tmp_path / "layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert torch.equal(q.u8_blob, q2.u8_blob)


def test_roundtrip_scale(tmp_path):
    q = _make_q4_0()
    path = tmp_path / "layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert torch.equal(q.qscheme.scale, q2.qscheme.scale)


def test_roundtrip_dequant_output(tmp_path):
    q = _make_q4_0()
    path = tmp_path / "layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert torch.allclose(q.to_base_tensor(), q2.to_base_tensor())


def test_roundtrip_type(tmp_path):
    q = _make_q4_0()
    path = tmp_path / "layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert type(q2) is QTensorTractScaleOnly


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


def test_nnef_name_preserved(tmp_path):
    q = _make_q4_0()
    q.nnef_name = "layers.0.weight"
    path = tmp_path / "layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert q2.nnef_name == "layers.0.weight"


def test_nnef_name_none_preserved(tmp_path):
    q = _make_q4_0()
    assert q.nnef_name is None
    path = tmp_path / "layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert q2.nnef_name is None


def test_group_size_preserved(tmp_path):
    q = _make_q4_0()
    path = tmp_path / "layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert q2.qscheme.group_size == q.qscheme.group_size


def test_n_bits_preserved(tmp_path):
    q = _make_q4_0()
    path = tmp_path / "layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert q2.qscheme.n_bits == q.qscheme.n_bits


def test_dequant_dtype_preserved(tmp_path):
    q = _make_q4_0()
    path = tmp_path / "layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert q2.dequant_to_dtype == q.dequant_to_dtype


def test_decompressed_shape_preserved(tmp_path):
    q = _make_q4_0(rows=64, cols=128)
    path = tmp_path / "layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert q2.decompressed_shape == q.decompressed_shape


# ---------------------------------------------------------------------------
# File is valid safetensors (not pickle)
# ---------------------------------------------------------------------------


def test_file_is_valid_safetensors(tmp_path):
    """Saved file must be readable as raw safetensors (no pickle involved)."""
    from safetensors import safe_open

    q = _make_q4_0()
    q.nnef_name = "proj.weight"
    path = tmp_path / "layer.safetensors"
    save_qtensor(q, path)

    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        meta = f.metadata()

    assert "u8_blob" in keys
    assert "scale" in keys
    assert meta.get("__qtensor_type__") == "QTensorTractScaleOnly"
    assert meta.get("nnef_name") == "proj.weight"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_save_unregistered_type_raises():
    class _FakeQTensor:
        pass

    with pytest.raises(ValueError, match="No QTensorSerde registered"):
        save_qtensor(_FakeQTensor(), "/tmp/dummy.safetensors")


def test_load_missing_type_key_raises(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "bad.safetensors"
    save_file({"data": torch.zeros(4)}, str(path))
    with pytest.raises(ValueError, match="__qtensor_type__"):
        load_qtensor(path)


def test_load_unregistered_type_raises(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "bad.safetensors"
    save_file(
        {"data": torch.zeros(4)},
        str(path),
        metadata={"__qtensor_type__": "UnknownType"},
    )
    with pytest.raises(ValueError, match="No QTensorSerde registered"):
        load_qtensor(path)


# ---------------------------------------------------------------------------
# Custom serde registration
# ---------------------------------------------------------------------------


def test_register_custom_serde():
    """A user-defined serde can be registered and used for save/load."""

    class _FakeQ:
        def __init__(self, data: torch.Tensor, scale: float) -> None:
            self.data = data
            self.scale = scale

    class _FakeQSerde(QTensorSerde):
        def type_name(self) -> str:
            return "_FakeQ"

        def to_tensors(self, q) -> dict[str, torch.Tensor]:
            return {"data": q.data}

        def to_metadata(self, q) -> dict[str, str]:
            return {"scale": str(q.scale)}

        def from_parts(self, tensors, metadata):
            return _FakeQ(tensors["data"], float(metadata["scale"]))

    register_qtensor_serde(_FakeQSerde())
    assert "_FakeQ" in _SERDE_REGISTRY

    # Cleanup to avoid polluting other tests.
    del _SERDE_REGISTRY["_FakeQ"]


def _q4_linear(device="cpu"):
    linear = torch.nn.Linear(64, 64, bias=False)
    weight = fp_to_tract_q4_0_with_min_max_calibration(
        linear.weight.detach().to(device)
    )
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    return linear


def test_qtensor_cpu_stays_quantized():
    from torch_to_nnef.tensor.quant import QTensor

    linear = _q4_linear()
    expected = linear.weight.decompress().clone()
    assert isinstance(linear.weight.cpu(), QTensor)
    linear.cpu()
    assert isinstance(linear.weight, QTensor)
    torch.testing.assert_close(linear.weight.decompress(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_module_cpu_and_cuda_move_quantized_weights():
    linear = _q4_linear("cuda")
    expected = linear.weight.decompress().cpu()
    linear.cpu()
    assert linear.weight.device.type == "cpu"
    assert linear.weight.u8_blob.device.type == "cpu"
    torch.testing.assert_close(linear.weight.decompress(), expected, rtol=0, atol=0)
    linear.cuda()
    assert linear.weight.u8_blob.device.type == "cuda"
    linear.cuda(0)
    assert linear.weight.u8_blob.device == torch.device("cuda", 0)


def test_deepcopy_keeps_every_layer_quantized_and_its_own():
    import copy

    from neuropress.pipeline import QuantizationPipeline
    from neuropress.recipe_config import QuantizationRecipeConfig

    estimators = [
        {
            "type": "vptq",
            "vector_len": 4,
            "num_centroids": 16,
            "max_n_iter": 5,
            "cluster_backend_name": "torch",
            "use_hessian_weighting": False,
        },
        {
            "type": "gptq",
            "group_size": 32,
            "nbits": 4,
            "symmetric": False,
            "weight_transform": "q4_1_packed",
        },
        {
            "type": "gptq",
            "group_size": 32,
            "nbits": 4,
            "symmetric": True,
            "weight_transform": "q4_0",
        },
    ]
    for estimator in estimators:
        torch.manual_seed(15)
        model = torch.nn.Sequential(
            *(torch.nn.Linear(64, 64, bias=False) for _ in range(8))
        ).eval()
        config = QuantizationRecipeConfig.model_validate(
            {"estimator": estimator, "kernel_lowering": None}
        )
        QuantizationPipeline(config.build()).run(model, [torch.randn(256, 64)])
        # Copies used to decompress temporaries whose ids deepcopy's memo then
        # reused, handing some layers another layer's weights; repeat to hit it.
        for _ in range(5):
            copied = copy.deepcopy(model)
            for original, clone in zip(model, copied, strict=True):
                assert type(clone.weight) is type(original.weight)
                assert clone.weight.u8_blob is not original.weight.u8_blob
                torch.testing.assert_close(
                    clone.weight.decompress(),
                    original.weight.decompress(),
                    rtol=0,
                    atol=0,
                )
        if estimator["type"] == "vptq":
            # A codebook shared by the group stays shared within the copy.
            centroids = [layer.weight.qscheme.centroids for layer in copied]
            assert (
                model[0].weight.qscheme.centroids is model[1].weight.qscheme.centroids
            )
            assert centroids[0] is centroids[1]
            assert centroids[0] is not model[0].weight.qscheme.centroids
