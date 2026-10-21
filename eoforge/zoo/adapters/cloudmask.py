"""OmniCloudMask (Wright et al.): pretrained cloud and cloud-shadow segmentation.

Input bands Red, Green, NIR (for Sentinel-2 the package itself uses B04, B03, B8A), any sensor
at 10-50 m. Weights download from Hugging Face on first use (the package handles that).
Classes: 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow.

Card params: patch_size (1000), patch_overlap (300), batch_size (1), dtype (fp32 | fp16 | bf16)
"""
from __future__ import annotations

import numpy as np

from .base import Adapter, Result, register


@register("omnicloudmask")
class OmniCloudMask(Adapter):
    requires = ("omnicloudmask",)

    def load(self):
        import omnicloudmask  # noqa: F401
        self.log(f"OmniCloudMask on {self.device}")

    def prefetch(self):
        from omnicloudmask.download_models import get_models
        get_models()          # the package downloads its weights here, not at import

    def predict(self, image, valid, profile, options=None):
        from omnicloudmask import predict_from_array
        p = {**self.card.params, **(options or {})}
        arr = image.copy()
        arr[:, ~valid] = 0                      # 0 is the package's no-data value
        dtype = {"fp32": "fp32", "fp16": "fp16", "bf16": "bf16"}[p.get("dtype", "fp32")]
        h, w = arr.shape[1:]
        patch = int(min(p.get("patch_size", 1000), max(h, w)))
        overlap = int(min(p.get("patch_overlap", 300), patch // 2))
        out = predict_from_array(arr, patch_size=patch, patch_overlap=overlap, batch_size=int(p.get("batch_size", 1)),
                                 inference_device=self.device, inference_dtype=dtype, export_confidence=True,
                                 softmax_output=True, no_data_value=0)
        out = np.asarray(out)
        if out.ndim == 3 and out.shape[0] == 4:
            probs = out.astype("float32")
            classes = probs.argmax(0).astype("uint8")
        else:                                   # older versions return class codes only
            probs, classes = None, out.reshape(h, w).astype("uint8")
        return Result(classes=classes, probs=probs)
