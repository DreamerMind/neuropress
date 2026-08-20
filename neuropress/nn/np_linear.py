"""NPLinear: drop-in replacement for ``nn.Linear`` with packed QTensor weights.

Five execution backends are available:

``"default"`` (or ``None`` / ``"torch"``)
    The QTensor's ``__torch_function__`` intercepts ``F.linear`` and calls
    ``decompress()`` (bit-unpack → per-group dequant) on **every** forward
    pass.  Zero migration cost, but adds ~0.3–1 ms per call for large weights.

``"compile"``
    A ``torch.compile``'d pure-torch dequantize-then-linear path.  torch's
    Inductor can fuse the nibble-unpack, per-group scale multiply, and matmul
    into a single optimised kernel.  Compiled once per process, cached across
    calls.  Works on CPU and CUDA.

``"cuda"``
    A hand-written CUDA kernel (``neuropress/kernels/q4_linear.cu``) that
    performs the full unpack + dequant + accumulate in one pass, without
    materialising the float weight matrix.  Lowest latency for decode on GPU.
    Requires a CUDA device and a CUDA toolkit for JIT compilation.

``"gemlite"``
    A Triton-based backend via the ``gemlite`` library (``pip install gemlite``).
    GemLite autotuned kernels cover group-quantized GEMM for nbits in
    ``{1, 2, 4, 8}`` and group sizes divisible by 32.  Competitive with
    hand-written CUDA kernels and easier to tune for non-standard shapes.
    Requires a CUDA device and ``gemlite`` installed.

``"auto"``
    Automatically selects the best available backend at construction time:

    1. ``"gemlite"`` — if ``gemlite`` is installed, a CUDA device is present,
       and the layer's ``(nbits, group_size)`` is GemLite-compatible.
    2. ``"cuda"``    — if a CUDA device is present.
    3. ``"compile"`` — otherwise.

    ``"auto"`` is never stored in :attr:`NPLinear.backend`; the resolved
    concrete backend is stored instead.

Supported QTensor types:

    - ``QTensorTractScaleOnly`` (Q4_0 symmetric)
    - ``QTensorScaleZp``        (Q4_1 asymmetric)

Usage::

    from neuropress.nn import NPLinear, linearize_qtensor_weights

    # Convert all nn.Linear with QTensor weights in a model:
    linearize_qtensor_weights(model, backend="auto")

    # Or wrap a single quantized linear:
    np_linear = NPLinear.from_linear(linear_with_qtensor_weight, backend="auto")

    # Benchmark / A-B test backends without re-quantizing:
    np_linear.set_backend("gemlite")
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backends import Backend, LinearBackend, is_cuda_jit_available

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _unpack_u8(u8_blob: torch.Tensor, nbits: int) -> torch.Tensor:
    """Unpack nibble-packed uint8 blob to a flat integer weight tensor.

    Args:
        u8_blob: Packed uint8 weights ``[N, K // (8 // nbits)]``.
        nbits: Bits per value — one of ``{1, 2, 4, 8}``.

    Returns:
        Integer weight tensor ``[N, K]`` with dtype ``torch.uint8``.
    """
    vpb = 8 // nbits
    N, K_packed = u8_blob.shape
    mask = (1 << nbits) - 1
    packed = u8_blob.to(torch.int32)
    shifts = torch.arange(vpb, device=u8_blob.device, dtype=torch.int32) * nbits
    return (
        ((packed.unsqueeze(-1) >> shifts) & mask)
        .reshape(N, K_packed * vpb)
        .to(torch.uint8)
    )


def _pack_u8(values: torch.Tensor, nbits: int) -> torch.Tensor:
    """Pack a one-value-per-byte uint8 tensor into a nibble-packed blob.

    Inverse of :func:`_unpack_u8`.

    Args:
        values: Unpacked uint8 weight values ``[N, K]``, each ``< 2**nbits``.
        nbits: Bits per value — one of ``{1, 2, 4, 8}``.

    Returns:
        Packed uint8 tensor ``[N, K // (8 // nbits)]``.
    """
    vpb = 8 // nbits
    N, K = values.shape
    grouped = values.to(torch.uint8).reshape(N, K // vpb, vpb)
    packed = torch.zeros(N, K // vpb, dtype=torch.uint8, device=values.device)
    for i in range(vpb):
        packed |= grouped[:, :, i] << (i * nbits)
    return packed


def _is_gemlite_compatible(nbits: int, group_size: int) -> bool:
    """Return ``True`` when GemLite can handle this ``(nbits, group_size)`` pair.

    Args:
        nbits: Bits per quantized value.
        group_size: Quantization group size.

    Returns:
        ``True`` if ``gemlite`` is importable, a CUDA device is present,
        *nbits* is in ``{1, 2, 4, 8}``, and *group_size* is divisible by 32.
    """
    if not torch.cuda.is_available():
        return False
    if nbits not in {1, 2, 4, 8}:
        return False
    if group_size % 32 != 0:
        return False
    try:
        import gemlite  # noqa: F401

        return True
    except ImportError:
        return False


def _resolve_backend(
    backend: str | LinearBackend | None,
    nbits: int | None = None,
    group_size: int | None = None,
) -> LinearBackend:
    """Resolve any backend alias to a concrete :class:`LinearBackend` member.

    ``LinearBackend.AUTO`` / ``"auto"`` selects the best available backend in
    priority order: GEMLITE (when compatible) → CUDA (JIT kernel if nvcc is
    available, otherwise eager PyTorch ops) → COMPILE (CPU-only fallback).
    ``None`` and ``"torch"`` are aliases for :attr:`LinearBackend.DEFAULT`.

    Args:
        backend: A :class:`LinearBackend` member, a recognised string alias,
            or ``None``.
        nbits: Bits per quantized value — used for GemLite compatibility
            check when *backend* is ``"auto"``.
        group_size: Quantization group size — used for GemLite compatibility
            check when *backend* is ``"auto"``.

    Returns:
        A concrete :class:`LinearBackend` member (never ``LinearBackend.AUTO``).

    Raises:
        ValueError: If *backend* is an unrecognised string.
    """
    if backend is None or backend == "torch":
        return LinearBackend.DEFAULT
    if isinstance(backend, LinearBackend | Backend):
        if backend.value != "auto":
            return LinearBackend(backend.value)
        backend = "auto"  # normalise enum → string and fall through
    if backend != "auto":
        return LinearBackend(backend)  # raises ValueError for unknown strings

    # AUTO resolution: GEMLITE > CUDA > COMPILE (CPU-only fallback).
    # On GPU, CUDA is preferred — it uses the JIT kernel when nvcc is
    # available, otherwise falls back to eager PyTorch ops (no torch.compile
    # warmup).  COMPILE is reserved for CPU where Inductor actually helps.
    if (
        nbits is not None
        and group_size is not None
        and _is_gemlite_compatible(nbits, group_size)
    ):
        return LinearBackend.GEMLITE
    if torch.cuda.is_available():
        return LinearBackend.CUDA
    return LinearBackend.COMPILE


def _torch_dtype_to_gemlite(dtype: torch.dtype):
    """Map a torch dtype to the corresponding gemlite DType."""
    from gemlite.core import DType  # type: ignore[import-untyped]

    return {
        torch.bfloat16: DType.BF16,
        torch.float16: DType.FP16,
        torch.float32: DType.FP32,
    }.get(dtype, DType.FP16)


def _build_gemlite_linear(
    u8_blob: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    bias: torch.Tensor | None,
    out_features: int,
    in_features: int,
    nbits: int,
    group_size: int,
    compute_dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """Construct and pack a ``GemLiteLinear`` layer from our tensor layout.

    Unpacks ``u8_blob`` to integer weights, reshapes scales and zero-points
    to GemLite's ``[N, K // group_size]`` layout, then calls
    ``GemLiteLinear.pack``.

    Args:
        u8_blob: Packed uint8 weights ``[out_features, in_features // (8 // nbits)]``.
        scale: Per-group float16 scales, 1-D, length
            ``out_features * (in_features // group_size)``.
        zero_point: Per-group float16 zero-points (same length as *scale*),
            or ``None`` for symmetric quantization.
        bias: Optional float bias ``[out_features]``.
        out_features: Output dimension N.
        in_features: Input dimension K.
        nbits: Bits per quantized value.
        group_size: Quantization group size.
        compute_dtype: Activation dtype the kernel will receive.  Must match
            the model's forward-pass dtype (e.g. ``torch.bfloat16`` for BF16
            models).  Defaults to ``torch.float16``.

    Returns:
        A packed ``GemLiteLinear`` module ready for inference.
    """
    from gemlite.core import GemLiteLinear  # type: ignore[import-untyped]

    gemlite_dtype = _torch_dtype_to_gemlite(compute_dtype)
    layer = GemLiteLinear(
        W_nbits=nbits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        input_dtype=gemlite_dtype,
        output_dtype=gemlite_dtype,
    )

    n_groups = in_features // group_size
    W_q = _unpack_u8(u8_blob, nbits)  # [N, K] uint8
    scales_2d = scale.to(torch.float16).reshape(out_features, n_groups)

    if zero_point is not None:
        zeros_2d = zero_point.to(torch.float16).reshape(out_features, n_groups)
    else:
        sym_zp = float(1 << (nbits - 1))
        zeros_2d = torch.full(
            (out_features, n_groups),
            sym_zp,
            dtype=torch.float16,
            device=u8_blob.device,
        )

    layer.pack(W_q, scales_2d, zeros_2d, bias)
    return layer


# ---------------------------------------------------------------------------
# Pure-torch dequant helper — used by the "compile" backend
# ---------------------------------------------------------------------------


def _q_dequant_linear(
    x: torch.Tensor,
    u8_blob: torch.Tensor,
    scale: torch.Tensor,
    nbits: int,
    group_size: int,
    zero_point: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Generic n-bit group-quantized dequant + linear (eager fallback).

    Supports any *nbits* in ``{1, 2, 4, 8}`` and both symmetric
    (``zero_point=None``) and asymmetric (``zero_point`` provided) schemes.

    Arithmetic is always performed in float32 to avoid fp16 constant-folding
    bugs in Inductor (``half_tensor - constant`` is not always lowered cleanly).
    The result is cast to ``x.dtype`` before the matmul.

    For ``torch.compile``, use the per-bitwidth specializations via
    :func:`_get_compiled` instead — they hardcode ``nbits`` so Dynamo never
    traces it as a symbolic ``s32``.

    Args:
        x: Input tensor ``[..., K]``.
        u8_blob: Packed uint8 weights ``[N, K // (8 // nbits)]`` (1-D flat is OK).
        scale: Per-group float16 scales ``[N * (K // group_size)]`` (1-D).
        nbits: Bits per value — one of ``{1, 2, 4, 8}``.
        group_size: Number of weight elements per group.
        zero_point: Per-group float16 zero-points (asymmetric), same shape as
            *scale*, or ``None`` for symmetric dequantization.
        bias: Optional float bias ``[N]``.

    Returns:
        Output tensor ``[..., N]`` matching ``x.dtype``.
    """
    vpb = 8 // nbits
    mask_val = (1 << nbits) - 1
    sym_zp_val = float(1 << (nbits - 1))

    N = u8_blob.shape[0]
    K = u8_blob.shape[1] * vpb
    n_gpb = K // group_size

    packed = u8_blob.to(torch.int32)
    shifts = torch.arange(vpb, device=u8_blob.device, dtype=torch.int32) * nbits
    vals = ((packed.unsqueeze(-1) >> shifts) & mask_val).reshape(N, K)

    vals_f = vals.to(torch.float32).reshape(N, n_gpb, group_size)
    sc = scale.to(torch.float32).reshape(N, n_gpb, 1)
    if zero_point is not None:
        zp = zero_point.to(torch.float32).reshape(N, n_gpb, 1)
        w = (vals_f - zp) * sc
    else:
        w = (vals_f - sym_zp_val) * sc

    return F.linear(x, w.reshape(N, K).to(x.dtype), bias)


# ---------------------------------------------------------------------------
# Per-bitwidth specializations for torch.compile
# ---------------------------------------------------------------------------
# Dynamo traces function arguments symbolically. When ``nbits`` is a parameter,
# expressions like ``(1 << nbits) - 1`` produce symbolic ``s32`` which
# Inductor cannot lower to integer constants, causing graph breaks.
#
# The fix: generate a separate function per supported bitwidth with all
# nbits-derived values baked in as Python-int literals. torch.compile sees
# only concrete constants — no symbolic tracing, no graph breaks.


def _make_dequant_linear(nbits: int):
    """Create a dequant+linear function with ``nbits`` baked in as a constant."""
    vpb = 8 // nbits
    mask = (1 << nbits) - 1
    sym_zp = float(1 << (nbits - 1))
    shift_step = nbits  # constant for arange multiplication

    def _dequant_linear(
        x: torch.Tensor,
        u8_blob: torch.Tensor,
        scale: torch.Tensor,
        group_size: int,
        zero_point: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        N = u8_blob.shape[0]
        K = u8_blob.shape[1] * vpb
        n_gpb = K // group_size

        packed = u8_blob.to(torch.int32)
        dev = u8_blob.device
        shifts = torch.arange(vpb, device=dev, dtype=torch.int32) * shift_step
        vals = ((packed.unsqueeze(-1) >> shifts) & mask).reshape(N, K)

        vals_f = vals.to(torch.float32).reshape(N, n_gpb, group_size)
        sc = scale.to(torch.float32).reshape(N, n_gpb, 1)
        if zero_point is not None:
            zp = zero_point.to(torch.float32).reshape(N, n_gpb, 1)
            w = (vals_f - zp) * sc
        else:
            w = (vals_f - sym_zp) * sc

        return F.linear(x, w.reshape(N, K).to(x.dtype), bias)

    return _dequant_linear


# Pre-built specializations — one per supported bitwidth.
_DEQUANT_FNS: dict[int, object] = {nb: _make_dequant_linear(nb) for nb in (1, 2, 4, 8)}

# Compiled singletons keyed by nbits — lazily created on first call.
_compiled_by_nbits: dict[int, object] = {}


def _get_compiled(nbits: int) -> object:
    """Return a compiled dequant+linear function for the given bitwidth."""
    fn = _compiled_by_nbits.get(nbits)
    if fn is None:
        fn = torch.compile(_DEQUANT_FNS[nbits])
        _compiled_by_nbits[nbits] = fn
    return fn


# ---------------------------------------------------------------------------
# NPLinear
# ---------------------------------------------------------------------------


class NPLinear(nn.Module):
    """Linear layer with packed n-bit quantized weight and pluggable backend.

    A drop-in replacement for ``nn.Linear`` whose ``weight`` is a packed
    QTensor (any ``nbits`` in ``{1, 2, 4, 8}``).  Supports both symmetric
    (scale-only) and asymmetric (scale + zero_point) group quantization.
    Select a backend at construction time; switch later with :meth:`set_backend`.

    Attributes:
        out_features: Output dimension N.
        in_features: Input dimension K.
        group_size: Quantization group size.
        nbits: Bits per quantized value (1, 2, 4, or 8).
        qtype: ``"sym"`` for symmetric or ``"asym"`` for asymmetric.
        backend: Active :class:`LinearBackend` (always a concrete member, never AUTO).
    """

    out_features: int
    in_features: int
    group_size: int
    nbits: int
    qtype: str
    backend: LinearBackend

    def __init__(
        self,
        u8_blob: torch.Tensor,
        scale: torch.Tensor,
        out_features: int,
        in_features: int,
        group_size: int,
        nbits: int,
        qtype: str,
        backend: str | LinearBackend | None,
        bias: torch.Tensor | None = None,
        zero_point: torch.Tensor | None = None,
        weight_q: torch.Tensor | None = None,
        qtensor_meta: dict[str, str] | None = None,
        compute_dtype: torch.dtype = torch.float16,
    ) -> None:
        """Initialise NPLinear.

        Args:
            u8_blob: Packed uint8 weight ``[out_features, in_features//2]``.
            scale: Per-group float16 scales, any shape (will be flattened to 1-D).
            out_features: Output dimension N.
            in_features: Input dimension K.
            group_size: Quantization group size.
            nbits: Bits per quantized value.
            qtype: ``"sym"`` (symmetric / scale-only) or
                ``"asym"`` (asymmetric / scale + zero_point).
            backend: ``"default"`` / ``None`` / ``"torch"``, ``"compile"``,
                ``"cuda"``, ``"gemlite"``, or ``"auto"`` (picks the best
                available backend; never stored as ``"auto"``).
            bias: Optional float bias ``[out_features]``.
            zero_point: Per-group float16 zero-points for Q4_1 (flattened to 1-D).
            weight_q: Original QTensor, required only for ``"default"`` backend.
            qtensor_meta: Serde metadata dict for QTensor reconstruction
                via :meth:`to_qtensor`.
            compute_dtype: Dtype the dequant/matmul runs in.  Clamped to
                ``float16`` for the GemLite backend, which has no float32 kernel.
        """
        super().__init__()
        self._qtensor_meta = qtensor_meta
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size
        self.nbits = nbits
        self.qtype = qtype
        self.backend = _resolve_backend(backend, nbits=nbits, group_size=group_size)
        # GemLite has no float32 kernel; use float16 when the weight was float32.
        if self.backend == LinearBackend.GEMLITE and compute_dtype == torch.float32:
            compute_dtype = torch.float16
        self._compute_dtype = compute_dtype

        if self.backend == LinearBackend.DEFAULT:
            if weight_q is None:
                raise ValueError("weight_q is required when backend='default'.")
            # Store QTensor directly — bypasses nn.Parameter type enforcement.
            object.__setattr__(self, "_weight_q", weight_q)
        else:
            self.register_buffer("u8_blob", u8_blob.contiguous())
            # Flatten scale/zero_point to 1-D for kernel indexing.
            self.register_buffer("scale", scale.reshape(-1).contiguous())
            if zero_point is not None:
                self.register_buffer("zero_point", zero_point.reshape(-1).contiguous())
            else:
                self.zero_point: torch.Tensor | None = None

            if self.backend == LinearBackend.GEMLITE:
                self._gemlite = _build_gemlite_linear(
                    u8_blob.contiguous(),
                    scale.reshape(-1).contiguous(),
                    zero_point.reshape(-1).contiguous()
                    if zero_point is not None
                    else None,
                    bias,
                    out_features,
                    in_features,
                    nbits,
                    group_size,
                    compute_dtype=compute_dtype,
                )

        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
        else:
            self.bias: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        backend: str | LinearBackend | None = LinearBackend.AUTO,
    ) -> NPLinear:
        """Build an ``NPLinear`` from an ``nn.Linear`` with a QTensor weight.

        Args:
            linear: An ``nn.Linear`` whose ``weight`` attribute is a
                ``QTensorTractScaleOnly`` (Q4_0) or ``QTensorScaleZp`` (Q4_1).
            backend: Target execution backend.  Defaults to ``LinearBackend.AUTO``
                (GemLite when compatible, CUDA when available, else compile).

        Returns:
            An ``NPLinear`` equivalent to *linear* using the specified backend.

        Raises:
            TypeError: If ``linear.weight`` is not a recognised QTensor type.
        """
        from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

        from ..transforms.weight_only import QTensorScaleZp

        weight_q = linear.weight
        bias = linear.bias.detach().clone() if linear.bias is not None else None

        # QTensorTractScaleOnly covers Q4_0 and Q2_0 (symmetric, any nbits).
        # QTensorScaleZp covers Q4_1 (asymmetric).
        if isinstance(weight_q, QTensorTractScaleOnly):
            qtype = "sym"
            zero_point = None
        elif isinstance(weight_q, QTensorScaleZp):
            qtype = "asym"
            zero_point = weight_q.qscheme.zero_point.detach().clone()
        else:
            raise TypeError(
                f"Unsupported weight type {type(weight_q).__name__!r}. "
                "Expected QTensorTractScaleOnly (symmetric) or "
                "QTensorScaleZp (asymmetric)."
            )

        bk = _resolve_backend(
            backend,
            nbits=weight_q.qscheme.n_bits,
            group_size=weight_q.qscheme.group_size,
        )

        # Capture serde metadata for later QTensor reconstruction.
        from ..t2n.qtensors import _SERDE_REGISTRY

        serde = _SERDE_REGISTRY[type(weight_q).__name__]
        qtensor_meta = serde.to_metadata(weight_q)

        return cls(
            u8_blob=weight_q.u8_blob.detach().clone(),
            scale=weight_q.qscheme.scale.detach().clone(),
            out_features=linear.out_features,
            in_features=linear.in_features,
            group_size=weight_q.qscheme.group_size,
            nbits=weight_q.qscheme.n_bits,
            qtype=qtype,
            backend=bk,
            bias=bias,
            zero_point=zero_point,
            weight_q=weight_q if bk == LinearBackend.DEFAULT else None,
            qtensor_meta=qtensor_meta,
            compute_dtype=weight_q.dequant_to_dtype,
        )

    # ------------------------------------------------------------------
    # QTensor reconstruction
    # ------------------------------------------------------------------

    def to_qtensor(self) -> torch.Tensor:
        """Reconstruct the original QTensor from this module's buffers.

        Uses the serde metadata captured at build time and the stored
        ``u8_blob``/``scale``/``zero_point`` buffers (reshaped back to the
        layout expected by :meth:`~neuropress.t2n.qtensors.QTensorSerde.from_parts`).

        Returns:
            A ``QTensorTractScaleOnly`` or ``QTensorScaleZp`` matching the
            original quantized weight.

        Raises:
            RuntimeError: If no serde metadata was stored (module not built
                via :meth:`from_linear`).
        """
        if self._qtensor_meta is None:
            raise RuntimeError(
                "Cannot reconstruct QTensor: no serde metadata stored. "
                "Build via NPLinear.from_linear() to enable to_qtensor()."
            )
        from ..t2n.qtensors import _SERDE_REGISTRY

        if self.backend == LinearBackend.DEFAULT:
            return self._weight_q

        type_name = "QTensorTractScaleOnly" if self.qtype == "sym" else "QTensorScaleZp"
        serde = _SERDE_REGISTRY[type_name]
        tensors: dict[str, torch.Tensor] = {
            "u8_blob": self.u8_blob,
            "scale": self.scale.reshape(-1, 1),
        }
        if self.zero_point is not None:
            tensors["zero_point"] = self.zero_point.reshape(-1, 1)
        return serde.from_parts(tensors, self._qtensor_meta)

    # ------------------------------------------------------------------
    # Backend switching
    # ------------------------------------------------------------------

    def set_backend(self, backend: str | LinearBackend | None) -> NPLinear:
        """Switch to a different backend in-place and return ``self``.

        ``"auto"`` re-resolves at call time using the layer's current
        ``(nbits, group_size)``, so calling this after moving the model
        to GPU will correctly pick ``"gemlite"`` or ``"cuda"``.

        Args:
            backend: New backend.  ``"auto"`` re-resolves to the best
                available concrete backend for this layer.

        Returns:
            ``self`` (for method chaining).

        Raises:
            RuntimeError: If switching to ``"default"`` when the original
                QTensor was not retained.
            RuntimeError: If switching to a kernel backend when ``u8_blob``
                is not available (layer was constructed with ``"default"``).
        """
        bk = _resolve_backend(backend, nbits=self.nbits, group_size=self.group_size)
        if bk == LinearBackend.DEFAULT and not hasattr(self, "_weight_q"):
            raise RuntimeError(
                "Cannot switch to LinearBackend.DEFAULT: the original QTensor was "
                "not retained.  Reconstruct NPLinear with backend='default'."
            )
        if bk != LinearBackend.DEFAULT and not hasattr(self, "u8_blob"):
            raise RuntimeError(
                f"Cannot switch to LinearBackend.{bk.name}: u8_blob is not "
                "available.  Reconstruct NPLinear with a non-default backend."
            )
        if bk == LinearBackend.GEMLITE and not hasattr(self, "_gemlite"):
            zp = getattr(self, "zero_point", None)
            self._gemlite = _build_gemlite_linear(
                self.u8_blob,
                self.scale,
                zp,
                getattr(self, "bias", None),
                self.out_features,
                self.in_features,
                self.nbits,
                self.group_size,
                compute_dtype=self._compute_dtype,
            )
        self.backend = bk
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transform with on-the-fly dequantization.

        Args:
            x: Input tensor ``[..., in_features]``.

        Returns:
            Output tensor ``[..., out_features]``.
        """
        if self.backend == LinearBackend.DEFAULT:
            return F.linear(x, self._weight_q, self.bias)
        if self.backend == LinearBackend.COMPILE:
            return self._forward_compile(x)
        if self.backend == LinearBackend.CUDA:
            return self._forward_cuda(x)
        if self.backend == LinearBackend.GEMLITE:
            return self._forward_gemlite(x)
        raise ValueError(f"Unknown backend: {self.backend}")  # pragma: no cover

    def _forward_compile(self, x: torch.Tensor) -> torch.Tensor:
        return _get_compiled(self.nbits)(
            x,
            self.u8_blob,
            self.scale,
            self.group_size,
            self.zero_point,
            self.bias,
        )

    def _forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if not is_cuda_jit_available():
            # Eager PyTorch dequant+matmul — no JIT or torch.compile overhead.
            return _q_dequant_linear(
                x,
                self.u8_blob,
                self.scale,
                self.nbits,
                self.group_size,
                self.zero_point,
                self.bias,
            )
        from .. import kernels

        # Flatten leading batch dims: [..., K] → [M, K].
        *leading, _ = x.shape
        orig_dtype = x.dtype
        if x.dtype == torch.bfloat16:
            x = x.half()
        x_2d = x.reshape(-1, self.in_features)
        y_2d = kernels.q_linear(
            x_2d,
            self.u8_blob,
            self.scale,
            self.nbits,
            self.group_size,
            self.zero_point,
            self.bias,
        )
        y = y_2d.reshape(*leading, self.out_features)
        return y.to(orig_dtype) if y.dtype != orig_dtype else y

    def _forward_gemlite(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        # Ensure the input dtype matches what GemLite was compiled for.
        # _compute_dtype is already clamped to fp16/bf16 at __init__ time,
        # but guard here as well for any serialised checkpoints with stale dtype.
        target = (
            self._compute_dtype
            if self._compute_dtype in (torch.float16, torch.bfloat16)
            else torch.float16
        )
        if x.dtype != target:
            x = x.to(target)
        out = self._gemlite(x)
        return out.to(orig) if out.dtype != orig else out

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Return a concise parameter summary for ``repr()``."""
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"nbits={self.nbits}, group_size={self.group_size}, "
            f"qtype={self.qtype}, backend={self.backend.value}"
        )


# ---------------------------------------------------------------------------
# Model-level transform
# ---------------------------------------------------------------------------


def linearize_qtensor_weights(
    model: nn.Module,
    backend: str | LinearBackend | None = LinearBackend.AUTO,
) -> nn.Module:
    """Replace all ``nn.Linear`` modules with QTensor weights with ``NPLinear``.

    Walks *model*, identifies every ``nn.Linear`` whose ``weight`` is a
    ``QTensorTractScaleOnly`` or ``QTensorScaleZp``, and replaces it with an
    :class:`NPLinear` using *backend*.  The model is mutated in-place.

    Each layer independently resolves ``"auto"`` based on its own
    ``(nbits, group_size)``, so mixed-compatibility models are handled
    correctly.

    Modules with unsupported weight types emit a warning and are left unchanged.

    Args:
        model: The ``nn.Module`` to transform.
        backend: Execution backend for all created ``NPLinear`` layers.
            Defaults to ``LinearBackend.AUTO`` (best available per layer).

    Returns:
        The same *model* (for chaining).

    Example::

        compress_fn(model, tokenizer=tokenizer)
        linearize_qtensor_weights(model, backend="auto")
        # model now runs inference without QTensor __torch_function__ overhead
    """
    from torch_to_nnef.tensor.quant.base import QTensor
    from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

    from ..transforms.weight_only import QTensorScaleZp

    _SUPPORTED = (QTensorTractScaleOnly, QTensorScaleZp)

    replacements: list[tuple[str, NPLinear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not isinstance(module.weight, QTensor):
            continue
        if not isinstance(module.weight, _SUPPORTED):
            # VQ or other QTensor types — handled by their own linearize pass.
            log.debug(
                "linearize_qtensor_weights: skipping %s (type=%s)",
                name,
                type(module.weight).__name__,
            )
            continue
        try:
            np_linear = NPLinear.from_linear(module, backend)
        except TypeError as exc:
            log.warning("linearize_qtensor_weights: skipping %s — %s", name, exc)
            continue
        replacements.append((name, np_linear))

    for name, np_linear in replacements:
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, attr = model, name
        setattr(parent, attr, np_linear)

    # Summarise which concrete backends were selected across all replaced layers.
    backend_counts: dict[str, int] = {}
    for _, np_linear in replacements:
        key = np_linear.backend.value
        backend_counts[key] = backend_counts.get(key, 0) + 1
    summary = ", ".join(f"{v}x{k}" for k, v in sorted(backend_counts.items()))
    log.info(
        "linearize_qtensor_weights: replaced %d layer(s) [%s]",
        len(replacements),
        summary or "none",
    )
    return model
