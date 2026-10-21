"""Download a handful of LandCoverNet Africa chips instead of the whole dataset.

LandCoverNet Africa v1.0 (Radiant Earth, CC-BY-4.0) is hosted on Source Cooperative as one
GeoTIFF per chip, date and band, so single chips can be fetched on their own:

    <tile>/<chip>/<tile>_<chip>_LC_10m.tif                       labels (band 1 class, band 2 consensus 0-100)
    <tile>/<chip>/S2/<tile>_<chip>_<YYYYMMDD>/<tile>_<chip>_<YYYYMMDD>_<band>_10m.tif

For each selected chip this script:
  1. reads the label file and keeps the most varied chip of a few candidates per tile,
  2. reads the scene classification layer (SCL) of every 2018 date and ranks dates by clear pixels,
  3. picks dates spread through each season (default: wet Nov-Apr and dry May-Oct), downloads
     10 bands and writes one cloud-masked median composite per season, stacked (20 bands),
  4. writes a label GeoTIFF (pixels below a consensus score are set to nodata).

Roughly 5-10 MB of downloads per chip.

    pip install boto3
    python tools/fetch_landcovernet.py --n 10 --out showcase/landcovernet
    python tools/run_showcase.py showcase/landcovernet
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

ENDPOINT = "https://data.source.coop"
BUCKET = "radiantearth"
ROOT = "landcovernet/landcovernet_af/data/v1.0/"

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]  # B01, B09 left out (atmospheric)
CLEAR_SCL = {4, 5, 6, 11}  # vegetation, bare soil, water, snow
CLASSES = {1: "water", 2: "artificial bare ground", 3: "natural bare ground", 4: "snow/ice",
           5: "woody", 6: "cultivated", 7: "semi-natural vegetation"}
TILE_RE = re.compile(r"^\d{2}[A-Z]{3}$")


# --------------------------------------------------------------------------- storage
class S3Store:
    """Anonymous, read-only access to Source Cooperative's S3-compatible API."""

    def __init__(self):
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
        except ImportError:
            sys.exit("This script needs boto3:  pip install boto3")
        self.s3 = boto3.client("s3", endpoint_url=ENDPOINT, region_name="us-west-2",
                               config=Config(signature_version=UNSIGNED, s3={"addressing_style": "path"},
                                             retries={"max_attempts": 5}))

    def list(self, prefix):
        """Return (sub-folders, files) directly under prefix."""
        dirs, files = [], []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
            dirs += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
            files += [o["Key"] for o in page.get("Contents", [])]
        return dirs, files

    def get(self, key) -> bytes:
        return self.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


class LocalStore:
    """A local copy with the same layout (for testing, or a partial download you already have)."""

    def __init__(self, root):
        self.root = Path(root)

    def list(self, prefix):
        p = self.root / prefix
        if not p.is_dir():
            return [], []
        dirs = sorted(str(c.relative_to(self.root)) + "/" for c in p.iterdir() if c.is_dir())
        files = sorted(str(c.relative_to(self.root)) for c in p.iterdir() if c.is_file())
        return dirs, files

    def get(self, key) -> bytes:
        return (self.root / key).read_bytes()


# --------------------------------------------------------------------------- helpers
def name(prefix):
    return prefix.rstrip("/").rsplit("/", 1)[-1]


def find_tiles(store, prefix=ROOT, depth=0):
    """Walk down from the dataset root until folders named like Sentinel-2 tiles (e.g. 35KNA) appear."""
    dirs, _ = store.list(prefix)
    tiles = [d for d in dirs if TILE_RE.match(name(d))]
    if tiles or depth >= 3:
        return tiles
    out = []
    for d in dirs:
        out += find_tiles(store, d, depth + 1)
    return out


def southern_africa_first(tiles, seed):
    """Order tiles so that southern Africa (UTM zones 33-36, latitude bands J-L, around KAZA) comes first."""
    rng = random.Random(seed)
    def is_south(t):
        n = name(t)
        return 33 <= int(n[:2]) <= 36 and n[2] in "JKL"
    south = [t for t in tiles if is_south(t)]
    rest = [t for t in tiles if not is_south(t)]
    rng.shuffle(south)
    rng.shuffle(rest)
    return south + rest


def read_tif(data: bytes):
    with rasterio.MemoryFile(data) as mem, mem.open() as src:
        return src.read(), src.profile.copy()


def class_entropy(labels):
    v = labels[labels > 0]
    if v.size == 0:
        return 0.0
    p = np.bincount(v, minlength=8)[1:] / v.size
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# --------------------------------------------------------------------------- discovery
# Files are found by listing and matching names, not by building paths, so small layout
# differences from the documentation (folder names, suffixes) do not break the download.
LABEL_RE = re.compile(r"_LC(_10m)?\.tif$", re.I)
S2_RE = re.compile(r"(^|[/_])S2([/_])", re.I)
BAND_RE = re.compile(r"(\d{8}).*_(B0[1-9]|B1[0-2]|B8A|SCL|CLD)(_10m)?\.tif$", re.I)


def walk(store, prefix, depth=4):
    """All file keys below prefix (up to `depth` folder levels)."""
    dirs, files = store.list(prefix)
    if depth > 0:
        with ThreadPoolExecutor(8) as ex:
            for sub in ex.map(lambda d: walk(store, d, depth - 1), dirs):
                files += sub
    return files


def chip_files(store, chip_prefix):
    """{'label': key, 'bands': {date: {band: key}}} for one chip folder."""
    files = walk(store, chip_prefix)
    labels = [f for f in files if LABEL_RE.search(f) and not BAND_RE.search(f)]
    bands: dict[str, dict[str, str]] = {}
    for f in files:
        m = BAND_RE.search(f)
        if m and S2_RE.search(f[len(chip_prefix):]):
            bands.setdefault(m.group(1), {})[m.group(2).upper()] = f
    return {"label": labels[0] if labels else None, "bands": bands, "n_files": len(files), "sample": files[:6]}


def pick_chip(store, tile_prefix, candidates, rng, log):
    chips = store.list(tile_prefix)[0]
    if not chips:
        log(f"  {name(tile_prefix)}: no chip folders")
        return None
    rng.shuffle(chips)
    best = None
    for c in chips[:candidates]:
        info = chip_files(store, c)
        if info["label"] is None:
            log(f"  {name(c)}: no label file among {info['n_files']} files, e.g. {info['sample']}")
            continue
        if not info["bands"]:
            log(f"  {name(c)}: no Sentinel-2 band files among {info['n_files']} files, e.g. {info['sample']}")
            continue
        try:
            arr, prof = read_tif(store.get(info["label"]))
        except Exception as e:
            log(f"  {name(c)}: cannot read {info['label']}: {e}")
            continue
        score = class_entropy(arr[0])
        if best is None or score > best[0]:
            best = (score, c, info, arr, prof)
    return best


def chip_id(tile_prefix, chip_prefix):
    tile, chip = name(tile_prefix), name(chip_prefix)
    m = re.search(r"(\d{2})$", chip)
    return f"{tile}_{m.group(1) if m else chip}"


# --------------------------------------------------------------------------- per chip
def parse_seasons(spec: str) -> dict[str, set[int]]:
    """'wet:11-4,dry:5-10' -> {'wet': {11, 12, 1, 2, 3, 4}, 'dry': {5, ..., 10}} (ranges may wrap the year)."""
    out = {}
    for part in spec.split(","):
        nm, rng = part.split(":")
        a, b = (int(x) for x in rng.split("-"))
        out[nm.strip()] = set(range(a, b + 1)) if a <= b else set(range(a, 13)) | set(range(1, b + 1))
    return out


def fetch_chip(store, cid, info, labels, prof, out_dir: Path, n_dates, min_consensus, workers, seasons):
    avail = {d: b for d, b in info["bands"].items() if "SCL" in b and all(x in b for x in BANDS)}
    if not avail:
        have = sorted({x for b in info["bands"].values() for x in b})
        raise RuntimeError(f"no date has SCL and all of {BANDS}; bands seen: {have}")
    dates = sorted(avail)

    def scl(date):
        try:
            return date, read_tif(store.get(avail[date]["SCL"]))[0][0]
        except Exception:
            return date, None

    with ThreadPoolExecutor(workers) as ex:
        scls = {d: a for d, a in ex.map(scl, dates) if a is not None}
    if not scls:
        raise RuntimeError("could not read any SCL file")
    clear = {d: float(np.isin(a, list(CLEAR_SCL)).mean()) for d, a in scls.items()}

    # Pick dates per season, spread through each season, so the composite carries the
    # phenology the annual labels were made from (cultivated vs natural vegetation vs woody).
    picks = {}
    for season, months in seasons.items():
        in_season = sorted(d for d in clear if int(d[4:6]) in months)
        good = [d for d in in_season if clear[d] >= 0.5]
        if not good:
            good = sorted(in_season, key=lambda d: -clear[d])[:n_dates]
        if not good:
            raise RuntimeError(f"no Sentinel-2 dates in the {season} season (months {sorted(months)})")
        idx = np.unique(np.linspace(0, len(good) - 1, min(n_dates, len(good))).round().astype(int))
        picks[season] = [good[i] for i in idx]
    chosen = sorted({d for ds in picks.values() for d in ds})

    def band(args):
        date, b = args
        a, p = read_tif(store.get(avail[date][b]))
        return date, b, a[0].astype("float32"), p

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(band, [(d, b) for d in chosen for b in BANDS]))
    h, w = labels.shape[1:]
    data, img_prof = {}, None
    for date, b, a, p in res:
        data[(date, b)] = a
        img_prof = img_prof or p

    composites, names, min_obs = [], [], {}
    for season, ds in picks.items():
        stack = np.full((len(ds), len(BANDS), h, w), np.nan, dtype="float32")
        for i, d in enumerate(ds):
            for j, b in enumerate(BANDS):
                stack[i, j] = data[(d, b)]
            stack[i][:, ~np.isin(scls[d], list(CLEAR_SCL))] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # pixels cloudy on every date
            comp = np.nanmedian(stack, axis=0)
        n_obs = np.isfinite(stack[:, 0]).sum(0)
        comp[:, n_obs == 0] = np.nan
        composites.append(comp)
        names += [f"{season}_{b}" if len(picks) > 1 else b for b in BANDS]
        min_obs[season] = int(n_obs.min())
    comp = np.concatenate(composites)
    invalid = np.isnan(comp).any(0)                        # no clear view in at least one season
    comp = np.nan_to_num(comp, nan=0).clip(1, 65535).astype("uint16")
    comp[:, invalid] = 0

    out_dir.mkdir(parents=True, exist_ok=True)
    img_path, lab_path = out_dir / f"{cid}_s2_2018.tif", out_dir / f"{cid}_labels.tif"
    ip = img_prof.copy()
    ip.update(driver="GTiff", count=len(names), dtype="uint16", nodata=0, compress="deflate")
    with rasterio.open(img_path, "w", **ip) as dst:
        dst.write(comp)
        for i, b in enumerate(names):
            dst.set_band_description(i + 1, b)
        dst.update_tags(source="LandCoverNet Africa v1.0 (Radiant Earth, CC-BY-4.0)",
                        composite="median of clear observations per season",
                        **{f"dates_{k}": ",".join(v) for k, v in picks.items()})
    lab = labels[0].copy()
    if labels.shape[0] > 1 and min_consensus > 0:
        lab[labels[1] < min_consensus] = 0
    lp = prof.copy()
    lp.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(lab_path, "w", **lp) as dst:
        dst.write(lab.astype("uint8"), 1)
    counts = {CLASSES.get(int(k), str(k)): int(v) for k, v in zip(*np.unique(lab[lab > 0], return_counts=True))}
    return {"chip": cid, "image": img_path.name, "labels": lab_path.name, "dates": picks,
            "clear_fraction": {d: round(clear[d], 3) for d in chosen}, "bands": names,
            "min_clear_obs": min_obs, "nodata_px": int(invalid.sum()), "class_pixels": counts, "crs": str(ip["crs"])}


def inspect(store, tile_prefix, max_items=6):
    """Print the layout below one tile: a few entries per level, down to the files."""
    print(f"Layout below {tile_prefix}")

    def show(prefix, indent, depth):
        dirs, files = store.list(prefix)
        for f in files[:max_items]:
            print(" " * indent + name(f))
        if len(files) > max_items:
            print(" " * indent + f"... {len(files)} files")
        for d in dirs[:2 if indent else 3]:
            print(" " * indent + name(d) + "/")
            if depth < 4:
                show(d, indent + 2, depth + 1)
        if len(dirs) > 3:
            print(" " * indent + f"... {len(dirs)} folders")

    show(tile_prefix, 2, 0)


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=10, help="number of chips (one per tile)")
    ap.add_argument("--out", default="showcase/landcovernet")
    ap.add_argument("--dates", type=int, default=3, help="dates per season, spread through the season")
    ap.add_argument("--seasons", default="wet:11-4,dry:5-10",
                    help="season windows as name:firstmonth-lastmonth,... (default southern Africa wet/dry); "
                         "'year:1-12' gives a single composite")
    ap.add_argument("--candidates", type=int, default=4, help="chips per tile to compare for class variety")
    ap.add_argument("--min-consensus", type=int, default=0, help="drop labels with consensus score below this (0-100)")
    ap.add_argument("--tiles", nargs="*", help="specific tiles, e.g. 35KNA 34KGD (default: southern Africa first)")
    ap.add_argument("--local", help="read from a local copy with the same layout instead of Source Cooperative")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inspect", action="store_true", help="print the folder layout of one tile and stop")
    a = ap.parse_args(argv)

    seasons = parse_seasons(a.seasons)
    store = LocalStore(a.local) if a.local else S3Store()
    out = Path(a.out)
    rng = random.Random(a.seed)
    print("Listing tiles ...", flush=True)
    try:
        tiles = find_tiles(store)
    except Exception as e:
        sys.exit(f"Could not list {ENDPOINT}/{BUCKET}/{ROOT}: {e}\n"
                 "Check your internet connection, or browse https://source.coop/radiantearth/landcovernet/landcovernet_af")
    if not tiles:
        top = store.list("landcovernet/landcovernet_af/")[0]
        sys.exit(f"No tile folders found under {ROOT}. Folders at the dataset root: {[name(t) for t in top]}.\n"
                 "The layout may have changed; set ROOT at the top of this script, or browse "
                 "https://source.coop/radiantearth/landcovernet/landcovernet_af")
    if a.tiles:
        tiles = [t for t in tiles if name(t) in set(a.tiles)]
    else:
        tiles = southern_africa_first(tiles, a.seed)
    if a.inspect:
        inspect(store, tiles[0])
        return
    print(f"{len(tiles)} tiles available; fetching {a.n} chips", flush=True)

    def log(msg):
        print(msg, flush=True)

    manifest = []
    for t in tiles:
        if len(manifest) >= a.n:
            break
        log(f"Tile {name(t)}")
        picked = pick_chip(store, t, a.candidates, rng, log)
        if picked is None:
            log(f"  skip {name(t)}: no usable chip")
            continue
        score, chip_prefix, files, labels, prof = picked
        if score < 0.3:      # nearly a single class: not much to show
            log(f"  skip {name(t)}: best chip is nearly one class (entropy {score:.2f})")
            continue
        cid = chip_id(t, chip_prefix)
        try:
            info = fetch_chip(store, cid, files, labels, prof, out, a.dates, a.min_consensus, a.workers, seasons)
        except Exception as e:
            log(f"  skip {cid}: {e}")
            continue
        info["class_entropy"] = round(score, 3)
        manifest.append(info)
        print(f"  [{len(manifest)}/{a.n}] {info['chip']}: " + ", ".join(f"{k} {len(v)} dates" for k, v in info["dates"].items()) + ", "
              f"classes {list(info['class_pixels'])}", flush=True)
        (out / "manifest.json").write_text(json.dumps({"class_map": CLASSES, "seasons": {k: sorted(v) for k, v in seasons.items()}, "chips": manifest}, indent=2))
    print(f"Done: {len(manifest)} chips in {out}")


if __name__ == "__main__":
    main()
