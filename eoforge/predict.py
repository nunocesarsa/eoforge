"""Tiled inference over a whole scene, written as georeferenced GeoTIFFs.

Overlapping tiles are blended with a smooth window so tile edges do not show,
and per-pixel uncertainty is reported as normalised Shannon entropy (0 = certain,
1 = all classes equally likely).
"""
from __future__ import annotations

import numpy as np
import rasterio
import torch


def _window_weights(size: int, overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones((size, size), "float32")
    ramp = np.ones(size, "float32")
    r = np.linspace(0.05, 1.0, overlap, dtype="float32")
    ramp[:overlap], ramp[-overlap:] = r, r[::-1]
    return np.outer(ramp, ramp)


def _starts(n: int, tile: int, step: int):
    if n <= tile:
        return [0]
    s = list(range(0, n - tile, step))
    s.append(n - tile)
    return s


@torch.no_grad()
def predict_scene(model, image: np.ndarray, valid: np.ndarray, num_classes: int,
                  tile=256, overlap=32, batch_size=8, device="cpu"):
    """image: (C, H, W) normalised. Returns probabilities (K, H, W) float32."""
    model.eval()
    c, h, w = image.shape
    tile = min(tile, max(h, w))
    step = max(1, tile - 2 * overlap)
    weight = _window_weights(tile, overlap)
    acc = np.zeros((num_classes, h, w), "float32")
    norm = np.zeros((h, w), "float32")

    windows = [(y, x) for y in _starts(h, tile, step) for x in _starts(w, tile, step)]
    for i in range(0, len(windows), batch_size):
        batch, metas = [], []
        for y, x in windows[i:i + batch_size]:
            chip = image[:, y:y + tile, x:x + tile]
            th, tw = chip.shape[1:]
            if (th, tw) != (tile, tile):
                pad = np.zeros((c, tile, tile), "float32")
                pad[:, :th, :tw] = chip
                chip = pad
            if not valid[y:y + th, x:x + tw].any():
                continue
            batch.append(chip)
            metas.append((y, x, th, tw))
        if not batch:
            continue
        probs = torch.softmax(model(torch.from_numpy(np.stack(batch)).to(device)), 1).cpu().numpy()
        for p, (y, x, th, tw) in zip(probs, metas):
            wgt = weight[:th, :tw]
            acc[:, y:y + th, x:x + tw] += p[:, :th, :tw] * wgt
            norm[y:y + th, x:x + tw] += wgt
    norm[norm == 0] = 1.0
    return acc / norm


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-8, 1.0)
    return (-(p * np.log(p)).sum(0) / np.log(probs.shape[0])).astype("float32")


def write_outputs(out_dir, profile: dict, probs: np.ndarray, valid: np.ndarray, classes: list[str],
                  write_entropy=True, write_probabilities=False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = {k: profile[k] for k in ("crs", "transform", "height", "width")}
    paths = {}

    cls = probs.argmax(0).astype("uint8")
    cls[~valid] = 255
    p = out_dir / "classes.tif"
    with rasterio.open(p, "w", driver="GTiff", count=1, dtype="uint8", nodata=255,
                       compress="deflate", tiled=True, **base) as dst:
        dst.write(cls, 1)
        dst.write_colormap(1, _colormap(len(classes)))
        dst.update_tags(**{f"class_{i}": c for i, c in enumerate(classes)})
    paths["classes"] = str(p)

    if write_entropy:
        ent = entropy(probs)
        ent[~valid] = np.nan
        p = out_dir / "entropy.tif"
        with rasterio.open(p, "w", driver="GTiff", count=1, dtype="float32", nodata=np.nan,
                           compress="deflate", tiled=True, **base) as dst:
            dst.write(ent, 1)
            dst.set_band_description(1, "normalised Shannon entropy (0 certain, 1 uniform)")
        paths["entropy"] = str(p)

    if write_probabilities:
        pr = probs.copy()
        pr[:, ~valid] = np.nan
        p = out_dir / "probabilities.tif"
        with rasterio.open(p, "w", driver="GTiff", count=len(classes), dtype="float32", nodata=np.nan,
                           compress="deflate", tiled=True, **base) as dst:
            dst.write(pr)
            for i, c in enumerate(classes):
                dst.set_band_description(i + 1, c)
        paths["probabilities"] = str(p)
    return paths


_PALETTE = [
    (31, 119, 180), (44, 160, 44), (214, 39, 40), (255, 127, 14), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
]


def _colormap(k):
    cmap = {i: (*_PALETTE[i % len(_PALETTE)], 255) for i in range(k)}
    cmap[255] = (0, 0, 0, 0)
    return cmap
