"""torch_to_nnef integration sub-package.

Exposes the compression registry (:mod:`.registry`), QTensor safetensors
serialization utilities (:mod:`.qtensors`), and n-bit packing helpers
(:mod:`.packing`).

Importing this package registers all weight transforms (``"q4_0"``,
``"identity"``) into
:data:`~neuropress.estimator_configs.WEIGHT_TRANSFORM_REGISTRY` as a side
effect via :mod:`neuropress.transforms`.
"""

from .. import transforms as _transforms  # noqa: F401  (registers weight transforms)
from . import _qtensor_patch as _qtensor_patch  # patches QTensor device handling
from .model_io import (
    QuantizationManifest,
    attach_manifest,
    load_quantized_model,
    save_quantized_model,
)
from .packing import (
    BitPacking,
    BitPacking1,
    BitPacking2,
    BitPacking4,
    BitPacking8,
)
from .qtensors import (
    QTensorSerde,
    load_qtensor,
    register_qtensor_serde,
    save_qtensor,
)
from .registry import (
    NEUROPRESS_COMPRESSION,
    NEUROPRESS_COMPRESSION_REGISTRY,
    apply_kernel_lowering,
    run_quantization_pipeline,
)

__all__ = [
    "NEUROPRESS_COMPRESSION",
    "NEUROPRESS_COMPRESSION_REGISTRY",
    "BitPacking",
    "BitPacking1",
    "BitPacking2",
    "BitPacking4",
    "BitPacking8",
    "QTensorSerde",
    "QuantizationManifest",
    "apply_kernel_lowering",
    "attach_manifest",
    "load_qtensor",
    "load_quantized_model",
    "register_qtensor_serde",
    "run_quantization_pipeline",
    "save_qtensor",
    "save_quantized_model",
]
