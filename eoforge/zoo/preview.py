"""PNG previews in EPSG:4326 for showing results on a web map."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

from .bands import match_bands

MAX_PX = 2048


def _to_wgs84(arr, profile, resampling, nodata):
    src_crs, src_tf = profile["crs"], profile["transform"]
    h, w = arr.shape[-2:]
    bounds = rasterio.transform.array_bounds(h, w, src_tf)
    tf, ww, hh = calculate_default_transform(src_crs, "EPSG:4326", w, h, *bounds)
    scale = max(1.0, max(ww, hh) / MAX_PX)
    ww, hh = int(ww / scale), int(hh / scale)
    tf = tf * tf.scale(scale, scale)
    a = arr[None] if arr.ndim == 2 else arr
    out = np.full((a.shape[0], hh, ww), nodata, dtype=a.dtype)
    for i in range(a.shape[0]):
        reproject(a[i], out[i], src_transform=src_tf, src_crs=src_crs, dst_transform=tf, dst_crs="EPSG:4326",
                  resampling=resampling, src_nodata=nodata, dst_nodata=nodata)
    west, north = tf.c, tf.f
    east, south = west + tf.a * ww, north + tf.e * hh
    return out, [[south, west], [north, east]]


def _save_png(rgba, path):
    from PIL import Image
    Image.fromarray(rgba.astype("uint8"), "RGBA").save(path)


def rgb_from_image(image_path, prefix=None):
    with rasterio.open(image_path) as src:
        desc = [d or "" for d in src.descriptions]
        idx, missing = match_bands(["B04", "B03", "B02"], desc, prefix)
        if missing:
            # prefer the dry season of seasonal stacks, else first three bands
            idx, missing = match_bands(["B04", "B03", "B02"], desc, "dry")
        if missing:
            idx = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
        a = src.read(idx).astype("float32")
        prof = src.profile.copy()
    valid = a.sum(0) > 0
    out = np.zeros(a.shape[1:] + (4,), "uint8")
    if valid.any():
        lo, hi = np.percentile(a[:, valid], (2, 98))
        rgb = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) ** 0.9
        out[..., :3] = (rgb.transpose(1, 2, 0) * 255).astype("uint8")
        out[..., 3] = np.where(valid, 255, 0)
    return out, prof


def write_previews(out: Path, image_path, card, r, valid, profile):
    meta = {}
    rgba, prof = rgb_from_image(image_path)
    arr, bounds = _to_wgs84(rgba.transpose(2, 0, 1), prof, Resampling.bilinear, 0)
    _save_png(arr.transpose(1, 2, 0), out / "preview_image.png")
    meta["image"] = {"file": "preview_image.png", "bounds": bounds}

    if r.kind == "objects":
        ids = r.classes.astype("int64")
        rng = np.random.default_rng(7)
        n = int(ids.max()) + 1
        lut = np.concatenate([[[0, 0, 0, 0]], np.c_[rng.integers(40, 255, (n, 3)), np.full(n, 255)]])[:n]
        col = lut[ids].astype("uint8")
        col[~valid] = 0
    else:
        lut = np.zeros((256, 4), "uint8")
        for c in card.classes:
            if 0 <= c.value < 256:
                h = c.colour.lstrip("#")
                lut[c.value] = [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [255]
        if not card.classes:
            rng = np.random.default_rng(3)
            lut[:255] = np.c_[rng.integers(40, 255, (255, 3)), np.full(255, 255)]
        cls = r.classes.astype("int64").clip(0, 255)
        col = lut[cls]
        col[~valid] = 0
    arr, bounds = _to_wgs84(col.transpose(2, 0, 1).astype("uint8"), profile, Resampling.nearest, 0)
    _save_png(arr.transpose(1, 2, 0), out / "preview_result.png")
    meta["result"] = {"file": "preview_result.png", "bounds": bounds}

    if r.probs is not None:
        from ..predict import entropy
        import matplotlib
        e = entropy(r.probs)
        e[~valid] = np.nan
        cm = matplotlib.colormaps["magma"]
        rgba = (cm(np.nan_to_num(e, nan=0)) * 255).astype("uint8")
        rgba[..., 3] = np.where(np.isfinite(e), 255, 0)
        arr, bounds = _to_wgs84(rgba.transpose(2, 0, 1), profile, Resampling.bilinear, 0)
        _save_png(arr.transpose(1, 2, 0), out / "preview_entropy.png")
        meta["entropy"] = {"file": "preview_entropy.png", "bounds": bounds}
    (out / "preview.json").write_text(json.dumps(meta))
