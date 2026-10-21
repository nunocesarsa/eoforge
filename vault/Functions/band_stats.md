---
tags: [function, raster]
module: eoforge.data.raster
---
# band_stats

> Per-band mean and standard deviation over valid pixels.

**Module:** [[eoforge.data.raster]] · **Source:** `eoforge/data/raster.py` · **Kind:** function

```python
band_stats(scene: Scene, max_pixels=2_000_000, seed=0) -> (mean, std)
```

## How it works

Subsamples to `max_pixels` for large scenes. A zero std is replaced by 1 to avoid division by zero.

## Used by

[[train]] · [[check_band_shift]]

## See also

[[Normalisation]]
