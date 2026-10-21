"""Agricultural field models: Fields of The World (FTW) and Delineate Anything.

Both follow the preprocessing of the FTW tools (github.com/fieldsoftheworld/ftw-baselines, MIT
licence): Sentinel-2 reflectance divided by 3000 and, for Delineate Anything, clipped to 0-1.
The polygon merging in `merge_polygons` is adapted from ftw_tools/inference/utils.py (MIT,
Copyright (c) 2024 kerner-lab).

FTW (adapter 'ftw'): semantic segmentation into background / field / field boundary from the
same four bands (B04, B03, B02, B08) at two dates: start and end of the growing season
(card params.windows: 2), or one date for the single-window model. Field polygons are made
from the field class (separated by the boundary class).
  params: checkpoint url in weights.url, resize_factor: 2, tile: 256, overlap: 16, scale: 3000,
          min_field_px: 4, write_vector: true

Delineate Anything (adapter 'delineate_anything'): YOLO instance segmentation of individual
fields from a true-colour image at 0.25-10 m (ultralytics, AGPL-3.0).
  params: tile: 256, overlap: 64, resize_factor: 2, conf: 0.05, iou: 0.3, max_det: 100,
          merge_iou: 0.3, merge_contain: 0.8, min_area_m2: 0, scale: 3000, write_vector: true
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..weights import fetch
from .base import Adapter, Result, register, starts, tiled_probs


# ------------------------------------------------------------------ helpers
def polygons_from_labels(labels: np.ndarray, transform, crs, min_px: int = 1):
    import geopandas as gpd
    from rasterio import features
    from shapely.geometry import shape
    geoms, ids = [], []
    for geom, val in features.shapes(labels.astype("int32"), mask=labels > 0, transform=transform):
        g = shape(geom)
        if g.area > 0:
            geoms.append(g)
            ids.append(int(val))
    gdf = gpd.GeoDataFrame({"id": ids}, geometry=geoms, crs=crs)
    if min_px > 1 and len(gdf):
        px = abs(transform.a * transform.e)
        gdf = gdf[gdf.geometry.area >= min_px * px]
    return gdf


def merge_polygons(polygons, iou_thresh=0.3, contain_thresh=0.8, grid_size=None):
    """Merge polygons that overlap (IoU) or contain one another; adapted from FTW (MIT)."""
    import geopandas as gpd
    import shapely
    gdf = polygons.copy().reset_index(drop=True)
    if len(gdf) < 2:
        return gdf
    gdf["geometry"] = gdf.geometry.apply(shapely.make_valid)
    pairs = gpd.sjoin(gpd.GeoDataFrame(geometry=gdf.geometry, crs=gdf.crs), gdf[["geometry"]],
                      how="inner", predicate="intersects")
    left = pairs.index.to_numpy()
    right = pairs["index_right"].to_numpy()
    m = left < right
    left, right = left[m], right[m]
    if left.size:
        geoms = gdf.geometry.to_numpy()
        L, R = geoms[left], geoms[right]
        inter = shapely.area(shapely.intersection(L, R))
        a, b = shapely.area(L), shapely.area(R)
        iou = inter / np.maximum(a + b - inter, 1e-12)
        cont = inter / np.maximum(np.minimum(a, b), 1e-12)
        keep = (iou >= iou_thresh) | (cont >= contain_thresh)
        left, right = left[keep], right[keep]
    parent = np.arange(len(gdf))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(left, right):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi
    gdf["_g"] = np.unique([find(i) for i in range(len(gdf))], return_inverse=True)[1]
    merged = gdf.groupby("_g")["geometry"].apply(
        lambda s: shapely.union_all(s.to_numpy(), grid_size=grid_size)).reset_index(drop=True)
    out = gpd.GeoDataFrame(geometry=merged, crs=gdf.crs).explode(index_parts=False, ignore_index=True)
    out = out.explode(index_parts=False, ignore_index=True)
    out = out[(out.geometry.geom_type == "Polygon") & out.geometry.is_valid & ~out.geometry.is_empty]
    return out.reset_index(drop=True)


def rasterise_ids(gdf, profile):
    from rasterio import features
    h, w = profile["height"], profile["width"]
    if not len(gdf):
        return np.zeros((h, w), "uint32")
    shapes = ((g, i + 1) for i, g in enumerate(gdf.geometry))
    return features.rasterize(shapes, out_shape=(h, w), transform=profile["transform"], fill=0, dtype="uint32")


def save_vector(gdf, out_dir, name):
    if out_dir is None or gdf is None:
        return None
    p = Path(out_dir) / name
    g = gdf.copy()
    if g.crs is not None and g.crs.is_projected:
        g["area_m2"] = g.geometry.area.round(1)
    g.to_file(p, driver="GPKG")
    return p.name


# ------------------------------------------------------------------ FTW
def load_ftw_checkpoint(path):
    """Rebuild an FTW model from its Lightning checkpoint (as ftw_tools.inference.models does)."""
    import segmentation_models_pytorch as smp
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in ckpt["state_dict"].items()}
    for k in ("criterion.weight", "ce_loss.weight"):
        sd.pop(k, None)
    kind = hp["model"]
    common = dict(encoder_name=hp.get("backbone"), encoder_weights=None, in_channels=hp["in_channels"],
                  classes=hp["num_classes"])
    if kind == "unet":
        model = smp.Unet(**common)
    elif kind == "unet_r":
        model = smp.Unet(decoder_channels=(16, 32, 64, 128, 256), **common)
    elif kind == "upernet":
        model = smp.UPerNet(**common)
    elif kind == "deeplabv3+":
        model = smp.DeepLabV3Plus(**common)
    else:
        raise ValueError(f"FTW model type '{kind}' is not supported here (supported: unet, unet_r, upernet, deeplabv3+)")
    model.load_state_dict(sd, strict=True)
    return model, hp


@register("ftw")
class FieldsOfTheWorld(Adapter):
    requires = ("segmentation_models_pytorch",)

    def load(self):
        path = fetch(self.card, self.log)["main"]
        self.model, self.hp = load_ftw_checkpoint(path)
        self.model = self.model.to(self.device).eval()
        self.log(f"FTW {self.hp.get('model')}/{self.hp.get('backbone')} ({self.hp['in_channels']} bands, "
                 f"{self.hp['num_classes']} classes) on {self.device}")

    @torch.no_grad()
    def predict(self, image, valid, profile, options=None):
        import torch.nn.functional as F
        p = {**self.card.params, **(options or {})}
        if image.shape[0] != self.hp["in_channels"]:
            raise ValueError(f"FTW model expects {self.hp['in_channels']} bands, got {image.shape[0]}")
        x = image.astype("float32") / float(p.get("scale", 3000))
        x[:, ~valid] = 0
        rf = int(p.get("resize_factor", 2))

        def run(batch):
            t = torch.from_numpy(batch).to(self.device)
            h, w = t.shape[-2:]
            if rf != 1:
                t = F.interpolate(t, scale_factor=rf, mode="bilinear", align_corners=False)
            out = torch.softmax(self.model(t).float(), 1)
            if rf != 1:
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
            return out.cpu().numpy()

        k = self.hp["num_classes"]
        probs = tiled_probs(run, x, valid, k, int(p.get("tile", 256)), int(p.get("overlap", 16)), int(p.get("batch", 4)),
                            multiple=32, progress=p.get("_progress"))
        classes = probs.argmax(0).astype("uint8")
        info = {}
        if p.get("write_vector", True):
            from scipy import ndimage
            fields, n = ndimage.label((classes == 1) & valid)
            gdf = polygons_from_labels(fields, profile["transform"], profile["crs"], int(p.get("min_field_px", 4)))
            info["fields"] = int(len(gdf))
            v = save_vector(gdf, p.get("_out_dir"), "fields.gpkg")
            if v:
                info["vector"] = v
        return Result(classes=classes, probs=probs, info=info)


# ------------------------------------------------------------------ Delineate Anything
@register("delineate_anything")
class DelineateAnything(Adapter):
    requires = ("ultralytics",)

    def load(self):
        from ultralytics import YOLO
        path = fetch(self.card, self.log)["main"]
        self.model = YOLO(str(path))
        self.model.to(self.device)
        try:
            self.model.fuse()
        except Exception:
            pass
        self.log(f"Delineate Anything ({Path(path).name}) on {self.device}")

    @torch.no_grad()
    def predict(self, image, valid, profile, options=None):
        import geopandas as gpd
        import pandas as pd
        import rasterio
        import shapely.geometry
        import torch.nn.functional as F
        p = {**self.card.params, **(options or {})}
        x = np.clip(image[:3].astype("float32") / float(p.get("scale", 3000)), 0, 1)
        x[:, ~valid] = 0
        c, h, w = x.shape
        rf = int(p.get("resize_factor", 2))
        tile = int(min(p.get("tile", 256), max(h, w)))
        unit = max(1, 32 // np.gcd(32, rf))                      # YOLO needs tile * rf divisible by 32
        tile = int(np.ceil(tile / unit) * unit)
        overlap = int(min(p.get("overlap", 64), tile // 4))
        step = max(1, tile - 2 * overlap)
        tf = profile["transform"]
        wins = [(y, xx) for y in starts(h, tile, step) for xx in starts(w, tile, step)]
        polys = []
        for i, (y0, x0) in enumerate(wins):
            chip = x[:, y0:y0 + tile, x0:x0 + tile]
            th, tw = chip.shape[1:]
            if not valid[y0:y0 + th, x0:x0 + tw].any():
                continue
            if (th, tw) != (tile, tile):
                chip = np.pad(chip, ((0, 0), (0, tile - th), (0, tile - tw)))
            t = torch.from_numpy(chip[None]).to(self.device)
            t = F.interpolate(t, size=(tile * rf, tile * rf), mode="bilinear", align_corners=False)
            res = self.model.predict(t, conf=float(p.get("conf", 0.05)), iou=float(p.get("iou", 0.3)),
                                     max_det=int(p.get("max_det", 100)), device=self.device, verbose=False)[0]
            if res.masks is None:
                continue
            res.masks.orig_shape = (tile, tile)          # mask coordinates in tile pixels, as FTW does
            for xy, conf in zip(res.masks.xy, res.boxes.conf.cpu().numpy()):
                if len(xy) < 3:
                    continue
                pts = [tf * (x0 + float(px), y0 + float(py)) for px, py in xy]
                g = shapely.geometry.Polygon(pts)
                if g.is_valid and g.area > 0:
                    polys.append((g, float(conf)))
            if p.get("_progress"):
                p["_progress"]((i + 1) / len(wins))
        crs = profile["crs"]
        gdf = gpd.GeoDataFrame({"confidence": [c_ for _, c_ in polys]}, geometry=[g for g, _ in polys], crs=crs)
        if len(gdf):
            gdf["geometry"] = gdf.geometry.apply(lambda g: shapely.geometry.Polygon(g.exterior))   # close holes
            gdf = merge_polygons(gdf, float(p.get("merge_iou", 0.3)), float(p.get("merge_contain", 0.8)))
            if p.get("min_area_m2") and crs is not None and crs.is_projected:
                gdf = gdf[gdf.geometry.area >= float(p["min_area_m2"])].reset_index(drop=True)
        ids = rasterise_ids(gdf, profile)
        info = {"fields": int(len(gdf))}
        v = save_vector(gdf, p.get("_out_dir"), "fields.gpkg") if p.get("write_vector", True) else None
        if v:
            info["vector"] = v
        return Result(classes=ids, kind="objects", info=info)
