---
tags: [function, zoo]
module: eoforge.zoo
---
# grid_at_resolution

> The grid of an image at another pixel size: same CRS and extent.

**Module:** [[eoforge.zoo]] · **Source:** `eoforge/zoo/run.py` · **Kind:** function

```python
grid_at_resolution(path, res) -> {crs, transform, width, height}
```

Used by [[run_model]] with `options['match_resolution']` when the image's pixel size differs from the card's `resolution_m` by more than 25%. The bands are then read through a warped view: averaged when going coarser (10 m Sentinel-2 to the 30 m HLS burn scar model), bilinear otherwise. Outputs are written on the new grid.
