"""Adapters register themselves on import. Optional dependencies are imported lazily inside each adapter."""
from .base import ADAPTERS, Adapter, Result, missing_packages, register  # noqa: F401
from . import onnx_seg, eoforge_ckpt, terratorch_seg, cloudmask, sam, terramind, fields  # noqa: F401
