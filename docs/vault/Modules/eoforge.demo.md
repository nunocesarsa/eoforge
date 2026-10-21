---
tags: [module]
---
# eoforge.demo

Synthetic Sentinel-2-like data so the pipeline can be tried without downloads. Used in [[Test - Synthetic demo]].

**Functions:** [[make_demo]]

**Helpers:**
- `make_scene(path, size=512, seed=0, brightness=1.0, origin=...)`: smooth random class patches (6 classes), typical spectra for B2 B3 B4 B8 B11 B12, per-patch variation, woodland texture, crop rows, noise and a nodata edge strip; written as uint16 GeoTIFF in EPSG:32734. Returns the truth array and transform.
- `make_labels(path, truth, transform, seed=0, n_polygons=120, n_points=150)`: small single-class polygons plus random points, like a field campaign; written as GeoPackage.
- `_smooth_field(rng, size, scale)`: Gaussian-filtered noise via FFT.
