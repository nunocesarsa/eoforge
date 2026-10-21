"""Download a small tutorial dataset that matches the models in the EOForge zoo.

    python tools/get_tutorial_data.py                       # farms and woodland north of Lusaka, Zambia
    python tools/get_tutorial_data.py --lat -1.95 --lon 30.06 --size-km 5 --year 2023     # your own place

No account needed. All sources are public:
  Sentinel-2 L2A    Earth Search STAC (Element84, AWS open data)  -> 4 scenes of one area, 12 bands, 10 m
  ESA WorldCover    2021 land cover, 10 m (AWS open data)         -> reference map on the same grid
  Sen1Floods11      2 hand-labelled flood chips (Google Cloud)    -> Sentinel-2 L1C + water labels

The four Sentinel-2 scenes are picked by their cloud mask (SCL) inside your area:
  start   clear, near the start of the season   (FTW window A, land cover)
  end     clear, at the end of the season       (FTW window B, land cover, fields)
  burn    clear, late dry season                (burn scars: fire season in southern Africa)
  cloudy  30-70 % clouds                        (cloud mask)
The +1000 offset of data processed since 2022 is removed, so the files are plain reflectance x 10000.

About 30-80 MB of downloads. Everything goes to --out (default: tutorial/) with TUTORIAL.md, which
lists the commands to run each model and how to check the results.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as warp_xy

STAC = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
# Earth Search asset key -> Sentinel-2 band name (B10 is not in L2A)
ASSETS = {"coastal": "B01", "blue": "B02", "green": "B03", "red": "B04", "rededge1": "B05", "rededge2": "B06",
          "rededge3": "B07", "nir": "B08", "nir08": "B8A", "nir09": "B09", "swir16": "B11", "swir22": "B12"}
CLOUD_SCL = {3, 8, 9, 10}            # cloud shadow, cloud medium, cloud high, thin cirrus
WORLDCOVER = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
WC_CLASSES = {10: ("tree cover", "#006400"), 20: ("shrubland", "#ffbb22"), 30: ("grassland", "#ffff4c"),
              40: ("cropland", "#f096ff"), 50: ("built-up", "#fa0000"), 60: ("bare / sparse", "#b4b4b4"),
              70: ("snow and ice", "#f0f0f0"), 80: ("water", "#0064c8"), 90: ("herbaceous wetland", "#0096a0"),
              95: ("mangroves", "#00cf75"), 100: ("moss and lichen", "#fae6a0")}
FLOODS_LIST = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"
FLOODS_FILE = "https://storage.googleapis.com/sen1floods11/{name}"
FLOODS_DIR = "v1.1/data/flood_events/HandLabeled/"

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")


def log(msg):
    print(msg, flush=True)


def http_json(url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json",
                                                          "User-Agent": "eoforge-tutorial"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


# --------------------------------------------------------------------------- area
def make_grid(lat, lon, size_km, res=10.0):
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    crs = rasterio.crs.CRS.from_epsg(epsg)
    (x,), (y,) = warp_xy("EPSG:4326", crs, [lon], [lat])
    n = int(round(size_km * 1000 / res))
    x0 = np.floor((x - n * res / 2) / res) * res
    y0 = np.ceil((y + n * res / 2) / res) * res
    grid = {"crs": crs, "transform": from_origin(x0, y0, res, res), "width": n, "height": n}
    xs, ys = [x0, x0 + n * res, x0, x0 + n * res], [y0, y0, y0 - n * res, y0 - n * res]
    lons, lats = warp_xy(crs, "EPSG:4326", xs, ys)
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    return grid, bbox


def read_on_grid(url, grid, resampling=Resampling.bilinear, dtype="float32"):
    with rasterio.open(url) as src:
        with WarpedVRT(src, crs=grid["crs"], transform=grid["transform"], width=grid["width"],
                       height=grid["height"], resampling=resampling) as vrt:
            return vrt.read(1).astype(dtype)


# --------------------------------------------------------------------------- Sentinel-2
def search(bbox, start, end, max_cloud=100, min_cloud=0):
    body = {"collections": [COLLECTION], "bbox": bbox, "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "limit": 100, "query": {"eo:cloud_cover": {"lte": max_cloud, "gte": min_cloud}}}
    items = http_json(STAC, body).get("features", [])
    # keep scenes whose footprint covers the whole area (not one that cuts it at a tile edge)
    out = []
    for it in items:
        b = it.get("bbox") or [-1e9, -1e9, 1e9, 1e9]
        if b[0] <= bbox[0] and b[1] <= bbox[1] and b[2] >= bbox[2] and b[3] >= bbox[3]:
            out.append(it)
    return out


def scl_stats(item, grid):
    scl = read_on_grid(item["assets"]["scl"]["href"], grid, Resampling.nearest, "uint8")
    nodata = float((scl == 0).mean())
    cloud = float(np.isin(scl, list(CLOUD_SCL)).mean())
    return cloud, nodata


def has_offset(item) -> bool:
    """True when the COGs still carry the +1000 BOA offset (processing baseline 04.00 and later)."""
    p = item["properties"]
    if p.get("earthsearch:boa_offset_applied") is True:
        return False
    for a in item["assets"].values():
        for rb in a.get("raster:bands", []) or []:
            if rb.get("offset") not in (None, 0, 0.0):
                return True
    try:
        return float(p.get("s2:processing_baseline", "0")) >= 4.0
    except ValueError:
        return False


def pick(bbox, grid, start, end, want="clear", tries=6):
    if want == "clear":
        items = sorted(search(bbox, start, end, max_cloud=40), key=lambda i: i["properties"]["eo:cloud_cover"])
    else:
        items = sorted(search(bbox, start, end, max_cloud=90, min_cloud=15),
                       key=lambda i: abs(i["properties"]["eo:cloud_cover"] - 45))
    if not items:
        return None, None
    best, best_score, best_stats = None, None, None
    for it in items[:tries]:
        try:
            cloud, nodata = scl_stats(it, grid)
        except Exception as e:                       # one unreadable scene should not stop the search
            log(f"    skip {it['id']}: {e}")
            continue
        if nodata > 0.02:
            continue
        score = cloud if want == "clear" else abs(cloud - 0.45)
        log(f"    {it['id']}: {cloud:.0%} cloud in the area")
        if best is None or score < best_score:
            best, best_score, best_stats = it, score, cloud
        if want == "clear" and cloud < 0.01:
            break
    return best, best_stats


def write_scene(item, grid, path: Path):
    off = has_offset(item)
    bands = []
    for key, name in ASSETS.items():
        a = read_on_grid(item["assets"][key]["href"], grid)
        bands.append(a)
    arr = np.stack(bands)
    valid = np.any(arr > 0, axis=0)
    # check the metadata against the data: with the offset even the darkest pixels sit above ~1000
    if valid.any():
        dark = min(float(np.percentile(b[valid], 0.5)) for b in arr)
        if dark < 900:
            off = False
        elif dark >= 1000 and float(item["properties"].get("s2:processing_baseline", "0") or 0) >= 4.0:
            off = True
    if off:
        arr = arr - 1000
    arr = np.where(valid, np.clip(arr, 1, 65535), 0).astype("uint16")
    prof = {"driver": "GTiff", "dtype": "uint16", "count": len(ASSETS), "nodata": 0, "compress": "deflate",
            "tiled": True, "blockxsize": 256, "blockysize": 256, **grid}
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr)
        for i, name in enumerate(ASSETS.values(), 1):
            dst.set_band_description(i, name)
        p = item["properties"]
        dst.update_tags(source=f"Earth Search {COLLECTION}", item=item["id"], datetime=p.get("datetime", ""),
                        processing_level="L2A", processing_baseline=str(p.get("s2:processing_baseline", "")),
                        offset_removed=str(off), units="surface reflectance x 10000")
    return off


# --------------------------------------------------------------------------- WorldCover
def worldcover(bbox, grid, path: Path):
    """ESA WorldCover on the area's grid; the 3 x 3 degree tiles that touch the area are mosaicked."""
    a = np.zeros((grid["height"], grid["width"]), "uint8")
    for lat0 in range(int(np.floor(bbox[1] / 3) * 3), int(np.floor(bbox[3] / 3) * 3) + 1, 3):
        for lon0 in range(int(np.floor(bbox[0] / 3) * 3), int(np.floor(bbox[2] / 3) * 3) + 1, 3):
            tile = f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"
            part = read_on_grid(WORLDCOVER.format(tile=tile), grid, Resampling.nearest, "uint8")
            a = np.where(a == 0, part, a)
    with rasterio.open(path, "w", driver="GTiff", dtype="uint8", count=1, nodata=0, compress="deflate", **grid) as dst:
        dst.write(a, 1)
        dst.write_colormap(1, {k: tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) + (255,) for k, (_, c) in WC_CLASSES.items()})
        dst.update_tags(source="ESA WorldCover 10 m 2021 v200 (CC-BY-4.0)",
                        **{f"class_{k}": n for k, (n, _) in WC_CLASSES.items()})
    vals, counts = np.unique(a[a > 0], return_counts=True)
    return {WC_CLASSES.get(int(v), (str(v),))[0]: round(float(100 * c / max(1, counts.sum())), 1) for v, c in zip(vals, counts)}


# --------------------------------------------------------------------------- Sen1Floods11
def floods(out: Path, countries, n):
    out.mkdir(parents=True, exist_ok=True)
    got = []
    for country in countries:
        q = urllib.parse.urlencode({"prefix": f"{FLOODS_DIR}S2Hand/{country}", "maxResults": 200})
        names = [o["name"] for o in http_json(f"{FLOODS_LIST}?{q}").get("items", []) if o["name"].endswith(".tif")]
        for name in names[: n - len(got)]:
            chip = Path(name).name.replace("_S2Hand.tif", "")
            for kind in ("S2Hand", "LabelHand"):
                src = f"{FLOODS_DIR}{kind}/{chip}_{kind}.tif"
                dst = out / f"{chip}_{kind}.tif"
                if not dst.exists():
                    urllib.request.urlretrieve(FLOODS_FILE.format(name=urllib.parse.quote(src)), dst)
            got.append(chip)
            log(f"  {chip}")
        if len(got) >= n:
            break
    return got


# --------------------------------------------------------------------------- tutorial text
def tutorial_md(out: Path, man: dict) -> str:
    s = man["scenes"]
    f = lambda k: s[k]["file"] if k in s else f"(no {k} scene found)"
    rel = out.name
    wc = ", ".join(f"{k} {v}%" for k, v in sorted(man.get("worldcover", {}).items(), key=lambda kv: -kv[1]))
    fl = man.get("floods", [])
    flood_chip = fl[0] if fl else None
    lines = [
        "# EOForge tutorial data", "",
        f"Area: {man['lat']}, {man['lon']}, {man['size_km']} km square, UTM {man['crs']}.", "",
        "| File | What it is | Date |", "|---|---|---|",
    ]
    for k, desc in [("start", "clear, start of season"), ("end", "clear, end of season"),
                    ("burn", "clear, late dry season (fires)"), ("cloudy", "partly cloudy")]:
        if k in s:
            lines.append(f"| `{s[k]['file']}` | Sentinel-2 L2A, 12 bands, {desc} | {s[k]['date']} |")
    lines += [f"| `worldcover_2021.tif` | ESA WorldCover 2021 reference ({wc}) | 2021 |"]
    if fl:
        lines += [f"| `sen1floods11/` | {len(fl)} flood chips: `*_S2Hand.tif` (L1C, 13 bands) + `*_LabelHand.tif` (1 water, 0 dry, -1 no data) | |"]
    lines += ["", "Start the app with `eoforge app --folder <this project>`; the files show up in the image list. "
              "Or use the commands below (run from the project folder, venv active). Results go to "
              f"`{rel}/eoforge_results/`. `python tools/compare_maps.py RESULT REFERENCE` prints how a result "
              "agrees with a reference map.", ""]
    steps = [
        ("1. Land cover with Dynamic World",
         f"eoforge zoo run dynamic-world {rel}/{f('start')}\n"
         f"python tools/compare_maps.py {rel}/eoforge_results/{Path(f('start')).stem}__dynamic-world/classes.tif {rel}/worldcover_2021.tif\n"
         f"eoforge zoo run dynamic-world {rel}/{f('end')}      # the same place in the dry season",
         "Dynamic World was trained on L1C; this is L2A, so expect some shift towards bare and shrub. "
         "Different class schemes: compare trees to tree cover, crops to cropland, water to water. "
         "Dynamic World labels each date: in the late dry season deciduous woodland is leafless and grass is dry, "
         "so most of the area becomes *bare* (78 % in the Zambia example). Compare both dates."),
        ("2. Land cover with TerraMind",
         f"eoforge zoo run terramind-lulc {rel}/{f('start')}\n"
         f"python tools/compare_maps.py {rel}/eoforge_results/{Path(f('start')).stem}__terramind-lulc/classes.tif {rel}/worldcover_2021.tif",
         "All 12 L2A bands are present, so no stand-ins. Generated ('mental image'): run twice with another seed to see what is stable."),
        ("3. Field boundaries with Fields of The World (two dates)",
         None,
         f"In the app: model *Fields of The World*, Start of season = `{f('start')}`, End of season = `{f('end')}`. "
         "Writes fields.gpkg. Open it in QGIS over the image."),
        ("4. Field outlines with Delineate Anything",
         f"eoforge zoo run delineate-anything {rel}/{f('end')}",
         "One polygon per field from the true-colour bands; compare with step 3."),
        ("5. Clouds with OmniCloudMask",
         f"eoforge zoo run omnicloudmask {rel}/{f('cloudy')}",
         "Compare with the image preview; Sentinel-2's own SCL cloud mask is a rough second opinion."),
        ("6. Burn scars with Prithvi",
         f"eoforge zoo run prithvi-burnscars {rel}/{f('burn')} --match-resolution",
         "Trained on 30 m HLS in the US: --match-resolution resamples to 30 m first. Dark patches in the late-dry-season image are usually recent burns."),
    ]
    if flood_chip:
        steps.append(("7. Water and floods with Prithvi (Sen1Floods11)",
                      f"eoforge zoo run prithvi-floods {rel}/sen1floods11/{flood_chip}_S2Hand.tif\n"
                      f"python tools/compare_maps.py {rel}/sen1floods11/eoforge_results/{flood_chip}_S2Hand__prithvi-floods/classes.tif {rel}/sen1floods11/{flood_chip}_LabelHand.tif",
                      "L1C with 13 unnamed bands: EOForge reads them in Sentinel-2 order. This is the model's own test data, so agreement should be high."))
    steps += [
        ("8. Objects with SAM 2",
         f"eoforge zoo run sam2 {rel}/{f('end')}",
         "At 10 m SAM outlines fields, water bodies and large patches. It is made for < 2 m imagery."),
        ("9. Train your own model on WorldCover labels",
         f"eoforge train {rel}/train_worldcover.yaml\n"
         f"eoforge zoo run {rel}/runs/worldcover_unet/best.pt {rel}/{f('burn')}",
         "Trains on the start and end images with WorldCover as labels, validated on spatial blocks, then maps a third date it never saw. "
         "A model trained on one small area is a demo, not a product: see the pooled showcase for transfer."),
    ]
    for title, cmd, note in steps:
        lines += [f"## {title}", ""]
        if cmd:
            lines += ["```bash", cmd, "```", ""]
        lines += [note, ""]
    lines += ["## Sources", "Sentinel-2: Copernicus/ESA via Earth Search (Element84). WorldCover: ESA, CC-BY-4.0. "
              "Sen1Floods11: Bonafilia et al. 2020, Cloud to Street."]
    return "\n".join(lines) + "\n"


def train_config(out: Path, man: dict):
    import yaml
    s = man["scenes"]
    scenes = [{"name": k, "image": s[k]["file"], "labels": "worldcover_2021.tif"} for k in ("start", "end") if k in s]
    present = man.get("worldcover", {})
    names = {k: n for k, (n, _) in WC_CLASSES.items() if present.get(n, 0) >= 0.5}   # skip classes with < 0.5 %
    cfg = {"name": "worldcover_unet", "output_dir": "runs", "data": {"source": "local"}, "scenes": scenes,
           "labels": {"class_map": names, "classes": list(names.values())},
           "model": {"arch": "unet", "params": {"base": 32, "depth": 3, "dropout": 0.2}},
           "train": {"chip_size": 64, "chips_per_epoch": 512, "batch_size": 32, "epochs": 20, "lr": 0.002,
                     "block_size": 64, "val_fraction": 0.25},
           "predict": {"tile_size": 256, "overlap": 32}}
    (out / "train_worldcover.yaml").write_text(
        "# Train a U-Net on the tutorial images with ESA WorldCover 2021 as labels.\n"
        "# Run from the project folder:  eoforge train tutorial/train_worldcover.yaml\n"
        + yaml.safe_dump(cfg, sort_keys=False))


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="tutorial")
    ap.add_argument("--lat", type=float, default=-14.95, help="centre latitude (default: Chisamba farms, Zambia)")
    ap.add_argument("--lon", type=float, default=28.35, help="centre longitude")
    ap.add_argument("--size-km", type=float, default=6.0, help="side of the square area (km)")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--start", default="03-01:04-30", help="start-of-season window, MM-DD:MM-DD")
    ap.add_argument("--end", default="06-15:08-31", help="end-of-season window")
    ap.add_argument("--burn", default="09-01:10-31", help="fire-season window")
    ap.add_argument("--cloudy", default="01-01:02-28", help="window to look for a partly cloudy scene")
    ap.add_argument("--no-floods", action="store_true", help="skip the Sen1Floods11 chips")
    ap.add_argument("--flood-countries", default="Nigeria,Ghana,Somalia", help="Sen1Floods11 events to take chips from")
    ap.add_argument("--flood-chips", type=int, default=2)
    ap.add_argument("--text-only", action="store_true", help="only rewrite TUTORIAL.md and the training config "
                    "from an earlier download (tutorial.json)")
    a = ap.parse_args(argv)

    out = Path(a.out).resolve()
    if a.text_only:
        man = json.loads((out / "tutorial.json").read_text())
        if man.get("scenes") and "worldcover" in man:
            train_config(out, man)
        (out / "TUTORIAL.md").write_text(tutorial_md(out, man))
        log(f"Rewrote {out / 'TUTORIAL.md'}")
        return
    out.mkdir(parents=True, exist_ok=True)
    grid, bbox = make_grid(a.lat, a.lon, a.size_km)
    log(f"Area {a.size_km} km around {a.lat}, {a.lon} -> {grid['width']} x {grid['height']} px, {grid['crs']}")
    man = {"lat": a.lat, "lon": a.lon, "size_km": a.size_km, "crs": str(grid["crs"]), "bbox": bbox, "scenes": {}}

    for key, win, want in [("start", a.start, "clear"), ("end", a.end, "clear"), ("burn", a.burn, "clear"),
                           ("cloudy", a.cloudy, "cloudy")]:
        m0, m1 = win.split(":")
        log(f"\n[{key}] Sentinel-2 L2A {a.year}-{m0} .. {a.year}-{m1}")
        try:
            item, cloud = pick(bbox, grid, f"{a.year}-{m0}", f"{a.year}-{m1}", want)
        except Exception as e:
            log(f"  search failed: {e}")
            continue
        if item is None:
            log("  no suitable scene; try another --year or window")
            continue
        date = item["properties"]["datetime"][:10].replace("-", "")
        path = out / f"s2_{key}_{date}.tif"
        log(f"  using {item['id']} ({cloud:.0%} cloud); downloading 12 bands ...")
        off = write_scene(item, grid, path)
        log(f"  wrote {path.name}" + ("  (+1000 offset removed)" if off else ""))
        man["scenes"][key] = {"file": path.name, "date": date, "item": item["id"], "cloud_in_area": round(cloud, 3),
                              "offset_removed": off}

    log("\n[reference] ESA WorldCover 2021")
    try:
        man["worldcover"] = worldcover(bbox, grid, out / "worldcover_2021.tif")
        log("  " + ", ".join(f"{k} {v}%" for k, v in sorted(man["worldcover"].items(), key=lambda kv: -kv[1])))
    except Exception as e:
        log(f"  failed: {e}")

    if not a.no_floods:
        log("\n[floods] Sen1Floods11 hand-labelled chips")
        try:
            man["floods"] = floods(out / "sen1floods11", a.flood_countries.split(","), a.flood_chips)
        except Exception as e:
            log(f"  failed: {e}")

    (out / "tutorial.json").write_text(json.dumps(man, indent=2))
    if man["scenes"] and "worldcover" in man:
        train_config(out, man)
    (out / "TUTORIAL.md").write_text(tutorial_md(out, man))
    if not man["scenes"]:
        sys.exit("No Sentinel-2 scene was downloaded; see the messages above.")
    log(f"\nDone. Data in {out}. Next steps: {out / 'TUTORIAL.md'}")


if __name__ == "__main__":
    main()
