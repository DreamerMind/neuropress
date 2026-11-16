"""Trellis-coded weights as QTensors, with their safetensors serde.

:class:`QTensorTrellis` holds a :class:`~neuropress.methods.trellis.TrellisWeight` as packed
tile words (its ``u8_blob``) and a :class:`TrellisScheme` (code, scales, signs). It dequantizes
on use like the other QTensors, and :func:`~neuropress.t2n.model_io.save_quantized_model` stores
it packed through the serde registered here.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import torch
from torch_to_nnef.tensor.quant.base import QScheme, QTensor

from ..methods.trellis import TrellisWeight
from ..methods.trellis_code import TrellisCode
from ..t2n.qtensors import QTensorSerde, register_qtensor_serde

FORMAT_VERSION = "1"


class TrellisScheme(QScheme):
    """Everything but the packed tiles of a :class:`TrellisWeight`."""

    def __init__(
        self,
        code: TrellisCode,
        row_scale: torch.Tensor,
        col_scale: torch.Tensor,
        sign_out: torch.Tensor,
        sign_in: torch.Tensor,
    ) -> None:
        self.code = code
        self.row_scale = row_scale
        self.col_scale = col_scale
        self.sign_out = sign_out
        self.sign_in = sign_in

    @property
    def shape(self) -> tuple[int, int]:
        """``(out, in)``."""
        return self.row_scale.numel(), self.col_scale.numel()

    def to_device(self, new_device: Any) -> TrellisScheme:
        """Move the scales, the signs and the codebook table."""
        super().to_device(new_device)
        if self.code.lut is not None:
            self.code = dataclasses.replace(self.code, lut=self.code.lut.to(new_device))
        return self

    def quantize_as_torch(self, fp_tensor: torch.Tensor) -> torch.Tensor:
        """Not supported: trellis codes come from :func:`~neuropress.methods.trellis.trellis_quantize`."""
        raise NotImplementedError("trellis weights are built from a TrellisWeight")

    def quantize_as_u8(self, fp_tensor: torch.Tensor) -> torch.Tensor:
        """Not supported, see :meth:`quantize_as_torch`."""
        raise NotImplementedError("trellis weights are built from a TrellisWeight")

    def weight(self, u8_blob: torch.Tensor) -> TrellisWeight:
        """The :class:`TrellisWeight` of *u8_blob*."""
        n, k = self.shape
        packed = u8_blob.view(torch.int16).reshape(
            n // 16, k // 16, self.code.tile_words
        )
        dev = u8_blob.device
        return TrellisWeight(
            self.code,
            packed,
            self.row_scale.to(dev),
            self.col_scale.to(dev),
            self.sign_out.to(dev),
            self.sign_in.to(dev),
        )

    def dequantize(
        self, u8_tensor: torch.Tensor, target_dtype: torch.dtype
    ) -> torch.Tensor:
        """Dense weight from the packed tiles."""
        return self.weight(u8_tensor).dequantize(target_dtype)

    def __repr__(self) -> str:
        """Concise description."""
        n, k = self.shape
        return f"TrellisScheme([{n}, {k}], bits={self.code.bits}, codebook={self.code.codebook})"


class QTensorTrellis(QTensor):
    """Weight stored as a bitshift trellis code (see :mod:`neuropress.methods.trellis`)."""

    qscheme: TrellisScheme
    # The adapter assigns it as the parameter as is (dtype checks would dequantize it)
    assign_as_parameter = True

    @classmethod
    def from_weight(
        cls, tw: TrellisWeight, dequant_to_dtype: torch.dtype = torch.float16
    ) -> QTensorTrellis:
        """Wrap a coded weight; no requantization."""
        scheme = TrellisScheme(
            tw.code, tw.row_scale, tw.col_scale, tw.sign_out, tw.sign_in
        )
        return cls.from_packed(
            tw.packed.contiguous().view(torch.uint8).reshape(-1),
            scheme,
            dequant_to_dtype,
        )

    @classmethod
    def from_packed(
        cls,
        u8_blob: torch.Tensor,
        scheme: TrellisScheme,
        dequant_to_dtype: torch.dtype = torch.float16,
    ) -> QTensorTrellis:
        """Build from packed bytes and a scheme."""
        # The wrapper lives on the codes' device: QTensor.to compares against it
        obj = torch.Tensor._make_subclass(cls, torch.empty(0, device=u8_blob.device))
        obj.u8_compressors = []
        obj.u8_blob = u8_blob
        obj.qscheme = scheme
        obj.dequant_to_dtype = dequant_to_dtype
        obj.nnef_name = None
        obj.requires_grad = False
        return obj

    @property
    def weight(self) -> TrellisWeight:
        """The coded weight."""
        return self.qscheme.weight(self.u8_blob)

    def to(self, *args: Any, **kwargs: Any) -> QTensorTrellis:  # type: ignore[override]
        """Move the codes and scheme, or change the dequantized dtype; never requantizes."""
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        device = self.u8_blob.device if device is None else torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        dtype = dtype or self.dequant_to_dtype
        if device == self.u8_blob.device and dtype == self.dequant_to_dtype:
            return self
        scheme = copy.copy(self.qscheme).to_device(device)
        out = self.from_packed(self.u8_blob.to(device), scheme, dtype)
        out.nnef_name = self.nnef_name
        return out

    def _rebuild_with(
        self, fp_tensor: torch.Tensor, qscheme: QScheme, dequant_to_dtype: torch.dtype
    ) -> QTensorTrellis:
        """Same codes under *qscheme* (device and dtype moves never requantize)."""
        assert isinstance(qscheme, TrellisScheme)
        return self.from_packed(self.u8_blob, copy.copy(qscheme), dequant_to_dtype)


class _QTensorTrellisSerde(QTensorSerde):
    """Serde for :class:`QTensorTrellis`.

    Tensors: ``packed`` (uint8 tile words), ``row_scale`` / ``col_scale`` (fp16),
    ``sign_out`` / ``sign_in`` (int8), ``lut`` (fp16, ``lut`` codebook only).
    Metadata: ``format``, ``bits``, ``codebook``, ``tile_len``, ``shape``, ``dequant_to_dtype``.
    """

    def type_name(self) -> str:
        """``"QTensorTrellis"``."""
        return "QTensorTrellis"

    def to_tensors(self, q: Any) -> dict[str, torch.Tensor]:
        """Packed tiles, scales, signs and the codebook table if any."""
        s = q.qscheme
        tensors = {
            "packed": q.u8_blob.cpu(),
            "row_scale": s.row_scale.cpu().clone(),
            "col_scale": s.col_scale.cpu().clone(),
            "sign_out": s.sign_out.cpu().clone(),
            "sign_in": s.sign_in.cpu().clone(),
        }
        if s.code.lut is not None:
            tensors["lut"] = s.code.lut.to(torch.float16).cpu().clone()
        return tensors

    def to_metadata(self, q: Any) -> dict[str, str]:
        """Code parameters and shape."""
        s = q.qscheme
        n, k = s.shape
        return {
            "format": FORMAT_VERSION,
            "bits": repr(float(s.code.bits)),
            "codebook": s.code.codebook,
            "tile_len": str(s.code.tile_len),
            "shape": f"{n},{k}",
            "dequant_to_dtype": str(q.dequant_to_dtype).replace("torch.", ""),
        }

    def from_parts(
        self, tensors: dict[str, torch.Tensor], metadata: dict[str, str]
    ) -> Any:
        """Rebuild the QTensor."""
        if metadata.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported trellis format {metadata.get('format')!r}")
        code = TrellisCode(
            float(metadata["bits"]),
            metadata["codebook"],  # type: ignore[arg-type]
            int(metadata["tile_len"]),
            tensors.get("lut"),
        )
        scheme = TrellisScheme(
            code,
            tensors["row_scale"],
            tensors["col_scale"],
            tensors["sign_out"],
            tensors["sign_in"],
        )
        n, k = (int(x) for x in metadata["shape"].split(","))
        if scheme.shape != (n, k):
            raise ValueError("trellis scales do not match the stored shape")
        return QTensorTrellis.from_packed(
            tensors["packed"],
            scheme,
            getattr(torch, metadata.get("dequant_to_dtype", "float16")),
        )


register_qtensor_serde(_QTensorTrellisSerde())
