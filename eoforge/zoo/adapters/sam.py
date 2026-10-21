"""Segment Anything via segment-geospatial (samgeo): outlines objects, without class names.

Works on any 3-band view of the image (default true colour). The adapter writes an 8-bit RGB
GeoTIFF of the selected bands, runs automatic mask generation, and returns one id per object.
Weights download on first use (Hugging Face / Meta).

Card params:
  model_id: sam2-hiera-large | sam2-hiera-base-plus | sam2-hiera-small | sam2-hiera-tiny
  version: 2 (SamGeo2) | 1 (SamGeo, model_type vit_h/vit_l/vit_b)
  stretch: [2, 98]          percentile stretch to 8-bit
  sam_kwargs: {points_per_side: 32, pred_iou_thresh: 0.8, stability_score_thresh: 0.9, min_mask_region_area: 50}
               (passed to SamGeo2: points_per_batch, box_nms_thresh, crop_n_layers, use_m2m, ...)
  write_vector: true        also save objects as GeoPackage
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import rasterio

from .base import Adapter, Result, register


@register("samgeo")
class SamGeoAdapter(Adapter):
    requires = ("samgeo", "sam2")

    def load(self):
        p = self.card.params
        kw = dict(p.get("sam_kwargs", {}))
        if int(p.get("version", 2)) == 2:
            from samgeo import SamGeo2
            self.sam = SamGeo2(model_id=p.get("model_id", "sam2-hiera-large"), automatic=True, device=self.device, **kw)
        else:
            from samgeo import SamGeo
            self.sam = SamGeo(model_type=p.get("model_type", "vit_h"), automatic=True, device=self.device,
                              sam_kwargs=kw or None)
        self.log(f"SAM ({p.get('model_id', p.get('model_type'))}) on {self.device}")

    def predict(self, image, valid, profile, options=None):
        p = {**self.card.params, **(options or {})}
        lo_p, hi_p = p.get("stretch", [2, 98])
        rgb = image[:3] if image.shape[0] >= 3 else np.repeat(image[:1], 3, 0)
        out = np.zeros(rgb.shape, "uint8")
        for i in range(3):
            v = rgb[i][valid]
            if v.size:
                lo, hi = np.percentile(v, (lo_p, hi_p))
                out[i] = (np.clip((rgb[i] - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype("uint8")
        out[:, ~valid] = 0
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "rgb.tif"
            prof = {k: profile[k] for k in ("crs", "transform", "height", "width")}
            with rasterio.open(src, "w", driver="GTiff", count=3, dtype="uint8", **prof) as dst:
                dst.write(out)
            masks = Path(tmp) / "masks.tif"
            self.sam.generate(str(src), output=str(masks), foreground=True, unique=True)
            with rasterio.open(masks) as m:
                ids = m.read(1).astype("uint32")
            info = {}
            if p.get("write_vector", True) and p.get("_out_dir"):
                try:
                    vec = Path(p["_out_dir"]) / "objects.gpkg"
                    self.sam.raster_to_vector(str(masks), str(vec))
                    info["vector"] = vec.name
                except Exception as e:
                    info["vector_error"] = str(e)
        return Result(classes=ids, kind="objects", info=info)
