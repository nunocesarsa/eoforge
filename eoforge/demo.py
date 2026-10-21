"""Synthetic Sentinel-2-like scene + labels, so the pipeline can be tried without data.

The scene is placed in UTM 34S (around the Zambezi region). Classes form smooth,
realistic-looking patches; each has a typical spectrum (B2 B3 B4 B8 B11 B12)
plus pixel noise and per-patch variation, so the task is not trivially separable.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

CLASSES = ["water", "grassland", "shrubland", "woodland", "bare", "cropland"]
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
# rough surface reflectance (x10000)
SPECTRA = np.array([
    [ 600,  700,  500,  300,  150,  100],   # water
    [ 700,  950, 1000, 2600, 2700, 1900],   # grassland
    [ 600,  850,  950, 2300, 2500, 1800],   # shrubland
    [ 400,  600,  500, 2900, 1700,  900],   # woodland
    [1200, 1500, 1900, 2400, 3300, 2900],   # bare
    [ 650,  900,  800, 3300, 2300, 1400],   # cropland
], dtype="float32")


def _smooth_field(rng, size, scale):
    noise = rng.standard_normal((size, size))
    f = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(f, f)
    kernel = np.exp(-(fx ** 2 + fy ** 2) * (scale ** 2) * 2 * np.pi ** 2)
    return np.real(np.fft.ifft2(np.fft.fft2(noise) * kernel))


def make_scene(path: Path, size=512, seed=0, brightness=1.0, origin=(550_000, 8_050_000)):
    rng = np.random.default_rng(seed)
    fields = np.stack([_smooth_field(rng, size, 18) for _ in CLASSES])
    fields[0] -= 0.6 * fields[0].std()                 # less water
    fields[4] -= 0.3 * fields[4].std()                 # less bare
    truth = fields.argmax(0).astype("uint8")

    img = SPECTRA[truth].transpose(2, 0, 1).copy()     # (6, H, W)
    # per-patch variation (low-frequency) + texture + pixel noise
    for b in range(img.shape[0]):
        img[b] *= 1 + 0.08 * _smooth_field(rng, size, 30) / (_smooth_field(rng, size, 30).std() + 1e-9) * 0.5
    texture = _smooth_field(rng, size, 1.5)
    texture /= texture.std()
    img[3] += 180 * texture * (truth == 3)             # woodland canopy texture in NIR
    img[3] += 120 * np.sign(np.sin(np.arange(size) / 2.0))[None, :] * (truth == 5)  # crop rows
    img += rng.normal(0, 120, img.shape)
    img = np.clip(img * brightness, 0, 10000)

    # a nodata strip on the edge, like a swath border
    img[:, :, -12:] = 0

    transform = from_origin(origin[0], origin[1], 10, 10)
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=len(BANDS),
                       dtype="uint16", crs="EPSG:32734", transform=transform, nodata=0,
                       compress="deflate") as dst:
        dst.write(img.astype("uint16"))
        for i, b in enumerate(BANDS):
            dst.set_band_description(i + 1, b)
    return truth, transform


def make_labels(path: Path, truth, transform, seed=0, n_polygons=120, n_points=150):
    """Sample labels like a field/photo-interpretation campaign: small polygons + points."""
    rng = np.random.default_rng(seed)
    size = truth.shape[0]
    rows = []
    usable = size - 16
    tries = 0
    while len([r for r in rows if r["geometry"].geom_type == "Polygon"]) < n_polygons and tries < 20000:
        tries += 1
        r, c = rng.integers(4, usable - 8, 2)
        s = int(rng.integers(3, 8))
        win = truth[r:r + s, c:c + s]
        if (win == win[0, 0]).all():
            x0, y0 = transform * (c, r)
            x1, y1 = transform * (c + s, r + s)
            rows.append({"class": CLASSES[win[0, 0]], "source": "polygon",
                         "geometry": box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))})
    for _ in range(n_points):
        r, c = rng.integers(2, usable, 2)
        x, y = transform * (c + 0.5, r + 0.5)
        rows.append({"class": CLASSES[truth[r, c]], "source": "point", "geometry": Point(x, y)})
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:32734")
    gdf.to_file(path, driver="GPKG")
    return gdf


def make_demo(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    truth, tf = make_scene(out / "scene_2023.tif", seed=1)
    np.save(out / "truth_2023.npy", truth)
    gdf = make_labels(out / "labels_2023.gpkg", truth, tf, seed=2)
    # A second scene (different landscape, slightly brighter) to test transfer with `eoforge predict`
    truth2, _ = make_scene(out / "scene_other.tif", seed=7, brightness=1.04, origin=(560_000, 8_050_000))
    np.save(out / "truth_other.npy", truth2)
    print(f"Demo data in {out}: scene_2023.tif, labels_2023.gpkg ({len(gdf)} features), scene_other.tif")
