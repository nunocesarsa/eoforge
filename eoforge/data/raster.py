"""Reading imagery, rasterising labels and computing normalisation stats."""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features

from ..config import DataConfig, LabelConfig


@dataclass
class Scene:
    image: np.ndarray        # (C, H, W) float32, nodata pixels set to 0 after normalisation
    valid: np.ndarray        # (H, W) bool
    profile: dict
    band_names: list[str]


def read_scene(path, cfg: DataConfig) -> Scene:
    with rasterio.open(path) as src:
        bands = cfg.bands or list(range(1, src.count + 1))
        img = src.read(bands).astype("float32")
        nodata = cfg.nodata if cfg.nodata is not None else src.nodata
        profile = src.profile.copy()
        names = cfg.band_names or [src.descriptions[b - 1] or f"B{b}" for b in bands]
    valid = np.all(np.isfinite(img), axis=0)
    # nodata: the configured or file value, else pixels that are 0 in every band (usual fill value)
    valid &= ~np.all(img == (nodata if nodata is not None else 0), axis=0)
    return Scene(img, valid, profile, list(names))


def band_stats(scene: Scene, max_pixels: int = 2_000_000, seed: int = 0):
    """Per-band mean/std over valid pixels (subsampled for large scenes)."""
    idx = np.flatnonzero(scene.valid.ravel())
    if idx.size > max_pixels:
        idx = np.random.default_rng(seed).choice(idx, max_pixels, replace=False)
    flat = scene.image.reshape(scene.image.shape[0], -1)[:, idx]
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    std[std == 0] = 1.0
    return mean.astype("float32"), std.astype("float32")


def normalise(img: np.ndarray, valid: np.ndarray, mean, std) -> np.ndarray:
    out = (img - mean[:, None, None]) / std[:, None, None]
    out[:, ~valid] = 0.0
    return out.astype("float32")


RASTER_SUFFIXES = (".tif", ".tiff", ".vrt", ".img", ".jp2")


def rasterise_labels(cfg: LabelConfig, profile: dict):
    """Put labels on the image grid. Returns (mask, class_names).

    Vector files are burned in; raster files (GeoTIFF etc.) are resampled.
    """
    if str(cfg.path).lower().endswith(RASTER_SUFFIXES):
        return _raster_labels(cfg, profile)
    gdf = gpd.read_file(cfg.path)
    if cfg.class_field not in gdf.columns:
        raise ValueError(f"Label field '{cfg.class_field}' not found. Columns: {list(gdf.columns)}")
    gdf = gdf[gdf.geometry.notna() & gdf[cfg.class_field].notna()]
    if cfg.exclude:
        gdf = gdf[~gdf[cfg.class_field].astype(str).isin({str(e) for e in cfg.exclude})]
    if gdf.empty:
        raise ValueError("No labels left after removing empty geometries and labels.exclude")
    if gdf.crs is None:
        raise ValueError("Label file has no CRS")
    gdf = gdf.to_crs(profile["crs"])

    values = gdf[cfg.class_field]
    if cfg.classes:
        classes = [str(c) for c in cfg.classes]
    else:
        classes = sorted({str(v) for v in values})
    lookup = {c: i for i, c in enumerate(classes)}
    missing = {str(v) for v in values} - set(lookup)
    if missing:
        raise ValueError(f"Label values not in labels.classes: {sorted(missing)}")

    geoms = gdf.geometry
    if cfg.point_buffer > 0:
        is_pt = geoms.geom_type.isin(["Point", "MultiPoint"])
        geoms = geoms.where(~is_pt, geoms.buffer(cfg.point_buffer))

    shapes = ((g, lookup[str(v)]) for g, v in zip(geoms, values))
    mask = features.rasterize(
        shapes,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=cfg.ignore_index,
        dtype="uint8",
        all_touched=True,  # so point labels hit at least one pixel
    )
    return mask, classes


def _raster_labels(cfg: LabelConfig, profile: dict):
    """Resample a label raster onto the image grid (nearest neighbour) and map codes to class ids.

    Codes are mapped with `labels.class_map` ({code: name}); without it, each code becomes a
    class named after its value. Pixels that are nodata, not in class_map, or whose name is in
    `labels.exclude` are ignored.
    """
    from rasterio.warp import Resampling, reproject

    sentinel = -999_999
    with rasterio.open(cfg.path) as src:
        if src.crs is None:
            raise ValueError("Label raster has no CRS")
        codes = np.full((profile["height"], profile["width"]), sentinel, dtype="int32")
        reproject(source=rasterio.band(src, cfg.band), destination=codes,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                  dst_transform=profile["transform"], dst_crs=profile["crs"], dst_nodata=sentinel,
                  resampling=Resampling.nearest)
    present = [int(v) for v in np.unique(codes) if v != sentinel]
    if cfg.class_map:
        cmap = {int(k): str(v) for k, v in cfg.class_map.items()}
    else:
        cmap = {v: str(v) for v in present}
    excluded = {str(e) for e in cfg.exclude}
    cmap = {k: v for k, v in cmap.items() if v not in excluded and str(k) not in excluded}
    if cfg.classes:
        classes = [str(c) for c in cfg.classes]
        unknown = set(cmap.values()) - set(classes)
        if unknown:
            raise ValueError(f"class_map names not in labels.classes: {sorted(unknown)}")
    else:
        classes = list(dict.fromkeys(cmap[k] for k in sorted(cmap)))
    lookup = {c: i for i, c in enumerate(classes)}
    mask = np.full(codes.shape, cfg.ignore_index, dtype="uint8")
    for code, name in cmap.items():
        mask[codes == code] = lookup[name]
    if (mask != cfg.ignore_index).sum() == 0:
        raise ValueError(f"No labelled pixels after mapping; codes present in raster: {present}")
    return mask, classes
