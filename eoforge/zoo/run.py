"""Run a model card on a GeoTIFF and write the results.

    from eoforge.zoo import run_model
    out = run_model("dynamic-world", "scene.tif")

Writes to <out_dir>/<image>__<model>/ :
    classes.tif         class codes (or object ids) with colour table
    entropy.tif         per-pixel uncertainty, when the model gives probabilities
    probabilities.tif   optional (options={'write_probabilities': True})
    summary.json        model, image, bands used, run time, class areas
    preview_*.png       map previews in EPSG:4326 (used by the app)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import rasterio

from .adapters import ADAPTERS, missing_packages
from .bands import guess_descriptions, match_bands, prefixes
from .card import ModelCard, get_card
from . import preview


class BandError(ValueError):
    def __init__(self, missing, descriptions):
        self.missing, self.descriptions = missing, descriptions
        super().__init__(f"Image has no band(s) {missing}. Image bands: {descriptions}. "
                         "Pass bands explicitly (e.g. bands=[2,3,4,8]).")


def load_adapter(card: ModelCard, device="auto", log=print):
    if card.adapter not in ADAPTERS:
        raise ValueError(f"Card '{card.id}' uses unknown adapter '{card.adapter}'. Available: {sorted(ADAPTERS)}")
    cls = ADAPTERS[card.adapter]
    miss = missing_packages(cls)
    if miss:
        hint = card.install or "pip install " + " ".join(miss)
        raise ImportError(f"'{card.name}' needs {', '.join(miss)}. Install with:  {hint}")
    a = cls(card, device, log)
    a.load()
    return a


def prefetch(card: ModelCard, log=print):
    """Download a model's weights now so the first run does not wait (used by `eoforge zoo download`)."""
    if card.adapter not in ADAPTERS:
        raise ValueError(f"unknown adapter '{card.adapter}'")
    cls = ADAPTERS[card.adapter]
    miss = missing_packages(cls)
    if miss:
        raise ImportError(f"needs {', '.join(miss)}: {card.install or 'pip install ' + ' '.join(miss)}")
    cls(card, "cpu", log).prefetch()


def grid_at_resolution(path, res: float) -> dict:
    """The grid of `path` resampled to `res` (same CRS and extent)."""
    from rasterio.transform import from_origin
    with rasterio.open(path) as src:
        b = src.bounds
        w, h = max(1, int(round((b.right - b.left) / res))), max(1, int(round((b.top - b.bottom) / res)))
        return {"crs": src.crs, "transform": from_origin(b.left, b.top, res, res), "width": w, "height": h}


def _read_one(path, required, bands, prefix, fallback, grid=None):
    """Read `required` bands (or explicit 1-based `bands`) from one image, optionally onto another grid."""
    from rasterio.vrt import WarpedVRT
    notes = []
    with rasterio.open(path) as src:
        desc = [d or "" for d in src.descriptions]
        if bands is None and required:
            desc, guess = guess_descriptions(desc, required)
            if guess:
                notes.append(guess)
        if bands is None:
            if required:
                pre = [q for q in prefixes(desc) if q]
                if prefix is None and len(pre) > 1:
                    prefix = pre[0]
                    notes.append(f"The image has several band sets ({', '.join(pre)}); used '{prefix}'. "
                                 "Choose another with prefix / 'Use bands from'.")
                bands, missing = match_bands(required, desc, prefix)
                for i, (need, b) in enumerate(zip(required, bands)):
                    if b is None and need in fallback:
                        fb, _ = match_bands([fallback[need]], desc, prefix)
                        if fb[0] is not None:
                            bands[i] = fb[0]
                            notes.append(f"{need} not in the image; used {fallback[need]} instead")
                missing = [r for r, b in zip(required, bands) if b is None]
                if missing:
                    raise BandError(missing, desc)
            else:
                bands = list(range(1, src.count + 1))
        if any(b is None or b < 1 or b > src.count for b in bands):
            raise ValueError(f"Band numbers {bands} outside 1..{src.count} in {Path(path).name}")
        same = grid is None or (src.crs == grid["crs"] and src.transform == grid["transform"]
                                and src.width == grid["width"] and src.height == grid["height"])
        if same:
            img = src.read(list(bands)).astype("float32")
            profile = src.profile.copy()
        else:
            coarser = abs(grid["transform"].a) > abs(src.transform.a) * 1.5
            rs = rasterio.enums.Resampling.average if coarser else rasterio.enums.Resampling.bilinear
            with WarpedVRT(src, crs=grid["crs"], transform=grid["transform"], width=grid["width"],
                           height=grid["height"], resampling=rs, src_nodata=src.nodata, nodata=src.nodata or 0) as vrt:
                img = vrt.read(list(bands)).astype("float32")
            notes.append(f"{Path(path).name} resampled to {abs(grid['transform'].a):g} m pixels on a common grid")
            profile = {**src.profile, **grid}
        nod = src.nodata
        res, crs = abs(profile["transform"].a), src.crs
    valid = np.all(np.isfinite(img), axis=0)
    # nodata: the file's own value, else pixels that are 0 in every band (the usual fill of mosaics/exports)
    valid &= ~np.all(img == (nod if nod is not None else 0), axis=0)
    names = [desc[b - 1] or f"band {b}" for b in bands]
    return img, valid, profile, list(bands), names, res, crs, notes


def read_input(image_path, card: ModelCard, bands=None, prefix=None, windows=None, resolution=None):
    """Read the model's input. Returns (image, valid, profile, bands_used, band_names, res, crs, notes).

    Models with params.windows = 2 (e.g. Fields of The World) need the same bands at two dates.
    `windows` is a list of {"image": path or None, "prefix": str or None, "bands": [..] or None};
    by default the first two band prefixes of the image are used (wet_, dry_ in seasonal stacks).
    """
    fallback = dict(card.params.get("band_fallback", {}))
    n = int(card.params.get("windows", 1))
    grid0 = grid_at_resolution(image_path, resolution) if resolution else None
    if n <= 1:
        img, valid, profile, used, names, res, crs, notes = _read_one(image_path, card.bands, bands, prefix, fallback, grid0)
    else:
        if windows is None:
            with rasterio.open(image_path) as src:
                pre = [p for p in prefixes([d or "" for d in src.descriptions]) if p]
            if len(pre) < n:
                raise ValueError(f"'{card.name}' needs the same bands at {n} dates. This image has "
                                 f"{'no' if not pre else 'only ' + str(len(pre))} dated band sets (like wet_B04, dry_B04); "
                                 "choose a second image or band set for the second date.")
            windows = [{"image": None, "prefix": p, "bands": None} for p in pre[:n]]
        parts, grid = [], grid0
        for w in windows:
            pth = w.get("image") or image_path
            part = _read_one(pth, card.bands, w.get("bands"), w.get("prefix"), fallback, grid)
            if grid is None:
                grid = {k: part[2][k] for k in ("crs", "transform", "width", "height")}
            parts.append((w, part))
        img = np.concatenate([p[0] for _, p in parts])
        valid = np.logical_and.reduce([p[1] for _, p in parts])
        profile, res, crs = parts[0][1][2], parts[0][1][5], parts[0][1][6]
        used = [b for _, p in parts for b in p[3]]
        names = [f"{w.get('prefix') or Path(w.get('image') or image_path).stem}:{nm}" for w, p in parts for nm in p[4]]
        notes = [nt for _, p in parts for nt in p[7]]
    img[:, ~valid] = 0
    return img, valid, profile, used, names, res, crs, notes


def check_units(img, valid, card: ModelCard, options: dict):
    """Bring the input to the units the model expects (Sentinel-2 style reflectance x 10000).

    - floats in 0-1 (reflectance, as exported by many tools) are multiplied by 10000;
    - options['offset'] (e.g. 1000) is subtracted: Sentinel-2 processed with baseline 04.00 or later
      (data from 25 Jan 2022 on) carries +1000 unless the provider removed it (harmonised collections);
    - a warning is given when the data look like they still carry that offset.
    Cards with params.input: any (SAM, OmniCloudMask, your trained models) are left untouched.
    """
    notes = []
    if card.params.get("input", "reflectance") != "reflectance" or not valid.any():
        return img, notes
    sample = img[:, valid]
    if sample.shape[1] > 200_000:
        sample = sample[:, np.random.default_rng(0).choice(sample.shape[1], 200_000, replace=False)]
    if np.percentile(sample, 99) <= 2.0:
        img = img * 10000.0
        sample = sample * 10000.0
        notes.append("Input looks like reflectance in 0-1; multiplied by 10000 to match the model.")
    off = float(options.get("offset") or 0)
    if off:
        img = np.clip(img - off, 0, None)
        notes.append(f"Subtracted the Sentinel-2 offset of {off:g}.")
    elif sample.shape[0] >= 2 and np.percentile(sample, 1, axis=1).min() >= 1000:
        notes.append("Every band is above 1000 even in its darkest pixels: this may be Sentinel-2 data with the "
                     "+1000 offset (processing baseline 04.00+, from 2022). If so, rerun with offset 1000.")
    return img, notes


def run_model(card: ModelCard | str, image_path, out_dir=None, bands=None, prefix=None, device="auto",
              options=None, adapter=None, log=print, progress=None, windows=None) -> dict:
    """Run a model card on an image.

    options: model params to override (tile, overlap, ...), plus
      offset: 1000             subtract the Sentinel-2 L2A offset (baseline 04.00+, data from 2022)
      match_resolution: true   resample the input to the model's pixel size first
      write_probabilities: true
    """
    card = get_card(card) if isinstance(card, str) else card
    image_path = Path(image_path)
    t0 = time.time()
    opts = dict(options or {})
    target = None
    if opts.get("match_resolution") and card.resolution_m:
        with rasterio.open(image_path) as s_:
            if s_.crs is not None and s_.crs.is_projected and abs(abs(s_.transform.a) - card.resolution_m) / card.resolution_m > 0.25:
                target = float(card.resolution_m)
    img, valid, profile, used, band_names, res, crs, notes = read_input(image_path, card, bands, prefix, windows, target)
    img, unit_notes = check_units(img, valid, card, opts)
    img[:, ~valid] = 0
    warnings = list(notes) + unit_notes
    if card.resolution_m and crs is not None and crs.is_projected and abs(res - card.resolution_m) / card.resolution_m > 0.25:
        warnings.append(f"Image pixel size is {res:g} m; the model was made for {card.resolution_m:g} m.")
    if crs is not None and not crs.is_projected:
        warnings.append("Image is in geographic coordinates; areas are approximate.")
    for w in warnings:
        log("WARNING " + w)
    log(f"{card.name} on {image_path.name}: {img.shape[2]}x{img.shape[1]} px, bands {used}")

    out = Path(out_dir) if out_dir else image_path.parent / "eoforge_results"
    out = out / f"{image_path.stem}__{card.id}"
    out.mkdir(parents=True, exist_ok=True)

    adapter = adapter or load_adapter(card, device, log)
    opts["_progress"], opts["_out_dir"] = progress, str(out)
    res_obj = adapter.predict(img, valid, profile, opts)
    for w in res_obj.info.pop("warnings", []):
        log("WARNING " + w)
        warnings.append(w)
    paths = write_result(out, card, res_obj, valid, profile, opts.get("write_probabilities", False))
    runtime = time.time() - t0
    summary = {
        "model": {"id": card.id, "name": card.name, "adapter": card.adapter, "source": card.source, "licence": card.licence},
        "image": str(image_path), "bands_used": used, "band_names": band_names,
        "device": adapter.device, "runtime_s": round(runtime, 1), "warnings": warnings,
        "pixel_size_m": res, "kind": res_obj.kind, "outputs": {k: Path(v).name for k, v in paths.items()},
        "areas": class_areas(card, res_obj, valid, res if (crs is not None and crs.is_projected) else None),
        **({"info": res_obj.info} if res_obj.info else {}),
        **({"options": {k: v for k, v in opts.items() if not k.startswith("_")}} if opts else {}),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        preview.write_previews(out, image_path, card, res_obj, valid, profile)
    except Exception as e:          # previews are a convenience; never fail a run on them
        log(f"(preview skipped: {e})")
    log(f"Done in {runtime:.1f} s -> {out}")
    summary["folder"] = str(out)
    return summary


def class_areas(card, r, valid, pixel_size):
    px_ha = (pixel_size ** 2) / 10_000 if pixel_size else None
    n_valid = int(valid.sum())
    if r.kind == "objects":
        ids = r.classes[valid]
        n = int(len(np.unique(ids[ids > 0])))
        cov = float((ids > 0).mean()) if ids.size else 0.0
        return {"objects": n, "covered_fraction": round(cov, 4)}
    vals, counts = np.unique(r.classes[valid], return_counts=True)
    rows = []
    for v, c in zip(vals, counts):
        rows.append({"value": int(v), "name": card.class_name(int(v)), "pixels": int(c),
                     "percent": round(float(100 * c / max(n_valid, 1)), 2),
                     **({"hectares": round(float(c * px_ha), 2)} if px_ha else {})})
    return sorted(rows, key=lambda x: -x["pixels"])


def write_result(out: Path, card: ModelCard, r, valid, profile, write_probs=False) -> dict:
    base = {k: profile[k] for k in ("crs", "transform", "height", "width")}
    paths = {}
    if r.kind == "objects":
        arr = r.classes.astype("uint32")
        arr[~valid] = 0
        with rasterio.open(out / "objects.tif", "w", driver="GTiff", count=1, dtype="uint32", nodata=0,
                           compress="deflate", tiled=True, **base) as dst:
            dst.write(arr, 1)
        paths["objects"] = out / "objects.tif"
    else:
        cls = r.classes.astype("uint8")
        cls[~valid] = 255
        with rasterio.open(out / "classes.tif", "w", driver="GTiff", count=1, dtype="uint8", nodata=255,
                           compress="deflate", tiled=True, **base) as dst:
            dst.write(cls, 1)
            if card.classes:
                cmap = {c.value: hex_rgba(c.colour) for c in card.classes if 0 <= c.value < 255}
                cmap[255] = (0, 0, 0, 0)
                dst.write_colormap(1, cmap)
                dst.update_tags(**{f"class_{c.value}": c.name for c in card.classes}, model=card.id)
        paths["classes"] = out / "classes.tif"
    if r.probs is not None:
        from ..predict import entropy
        ent = entropy(r.probs)
        ent[~valid] = np.nan
        with rasterio.open(out / "entropy.tif", "w", driver="GTiff", count=1, dtype="float32", nodata=np.nan,
                           compress="deflate", tiled=True, **base) as dst:
            dst.write(ent, 1)
        paths["entropy"] = out / "entropy.tif"
        if write_probs:
            pr = r.probs.astype("float32").copy()
            pr[:, ~valid] = np.nan
            with rasterio.open(out / "probabilities.tif", "w", driver="GTiff", count=pr.shape[0], dtype="float32",
                               nodata=np.nan, compress="deflate", tiled=True, **base) as dst:
                dst.write(pr)
                for i, c in enumerate(card.classes[:pr.shape[0]]):
                    dst.set_band_description(i + 1, c.name)
            paths["probabilities"] = out / "probabilities.tif"
    for name, arr in r.extra.items():
        a = np.asarray(arr, "float32")
        a = a[None] if a.ndim == 2 else a
        with rasterio.open(out / f"{name}.tif", "w", driver="GTiff", count=a.shape[0], dtype="float32",
                           compress="deflate", tiled=True, **base) as dst:
            dst.write(a)
        paths[name] = out / f"{name}.tif"
    return paths


def hex_rgba(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
