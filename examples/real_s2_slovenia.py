"""Build a real-data test case from the eo-learn example patch.

Source: github.com/sentinel-hub/eo-learn, example_data/TestEOPatch
  - Sentinel-2 L1C, 13 bands, 68 dates (Jul 2015 - Dec 2017), 10 m, ~1 km2 in Slovenia (EPSG:32633)
  - cloud mask per date (CLM) and land-use reference polygons (LULC.gpkg, Slovenian
    agricultural land-use register, mapped 2017-2018)

Writes cloud-masked median composites of the growing season (April-October) for
2016 and 2017 as GeoTIFFs, plus the reference polygons as a GeoPackage.

    git clone --depth 1 --filter=blob:none --sparse https://github.com/sentinel-hub/eo-learn.git
    cd eo-learn && git sparse-checkout set example_data && cd ..
    python examples/real_s2_slovenia.py eo-learn/example_data/TestEOPatch real_data
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]


def composite(bands, clm, times, year, months=range(4, 11), min_clear=0.5):
    """Median of cloud-free observations in the given year and months."""
    sel = np.array([t.year == year and t.month in months for t in times])
    clear_frac = 1 - clm[..., 0].mean(axis=(1, 2))
    sel &= clear_frac >= min_clear                      # drop mostly cloudy scenes
    stack = bands[sel].copy()                           # (T, H, W, 13)
    stack[clm[sel, ..., 0].astype(bool)] = np.nan       # mask cloudy pixels
    comp = np.nanmedian(stack, axis=0)                  # (H, W, 13)
    n_obs = np.isfinite(stack[..., 0]).sum(0)
    return comp, int(sel.sum()), n_obs


def main(patch: Path, out: Path):
    from datetime import datetime
    out.mkdir(parents=True, exist_ok=True)
    bands = np.load(patch / "data/BANDS-S2-L1C.npy")          # (T, H, W, 13) reflectance 0-1
    clm = np.load(patch / "mask/CLM.npy")                      # (T, H, W, 1) 1 = cloud
    times = [datetime.fromisoformat(t) for t in json.load(open(patch / "timestamps.json"))]
    bbox = json.load(open(patch / "bbox.geojson"))
    xs = [c[0] for c in bbox["coordinates"][0]]
    ys = [c[1] for c in bbox["coordinates"][0]]
    _, h, w, _ = bands.shape
    transform = from_bounds(min(xs), min(ys), max(xs), max(ys), w, h)
    crs = "EPSG:32633"

    summary = {}
    for year in (2016, 2017):
        comp, n_dates, n_obs = composite(bands, clm, times, year)
        arr = np.round(np.nan_to_num(comp, nan=0) * 10000).clip(1, 65535).astype("uint16")
        arr[n_obs == 0] = 0                                  # nodata where never clear
        path = out / f"s2_slovenia_{year}_AprOct.tif"
        with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=13, dtype="uint16",
                           crs=crs, transform=transform, nodata=0, compress="deflate") as dst:
            dst.write(arr.transpose(2, 0, 1))
            for i, b in enumerate(S2_BANDS):
                dst.set_band_description(i + 1, b)
            dst.update_tags(source="eo-learn TestEOPatch (Sentinel-2 L1C)", composite=f"median {year} Apr-Oct",
                            n_dates=str(n_dates))
        summary[year] = {"file": path.name, "dates_used": n_dates, "min_clear_obs_per_px": int(n_obs.min())}
        print(f"{path.name}: {n_dates} dates, clear obs per pixel {n_obs.min()}-{n_obs.max()}")

    gdf = gpd.read_file(patch / "vector_timeless/LULC.gpkg").to_crs(crs)
    gdf = gdf[["LULC_ID", "LULC_NAME", "DATE", "geometry"]]
    gdf["LULC_NAME"] = gdf["LULC_NAME"].replace({"schrubland": "shrubland"})
    gdf.to_file(out / "lulc_reference.gpkg", driver="GPKG")
    # Also keep the reference raster from eo-learn to check our georeferencing.
    np.save(out / "lulc_reference_raster.npy", np.load(patch / "mask_timeless/LULC.npy")[..., 0])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"lulc_reference.gpkg: {len(gdf)} polygons; classes {sorted(gdf.LULC_NAME.unique())}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
