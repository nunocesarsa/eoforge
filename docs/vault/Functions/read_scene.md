---
tags: [function, raster]
module: eoforge.data.raster
---
# read_scene

> Read the selected bands of a GeoTIFF and work out which pixels are valid.

**Module:** [[eoforge.data.raster]] · **Source:** `eoforge/data/raster.py` · **Kind:** function

```python
read_scene(path, cfg: DataConfig) -> Scene
```

## Parameters

| Name | Type | Meaning |
|---|---|---|
| `path` | `str | Path` | GeoTIFF |
| `cfg` | `DataConfig` | `bands`, `band_names`, `nodata` |

## Returns

`Scene(image, valid, profile, band_names)`: image as float32 (C, H, W).

## How it works

A pixel is invalid if any band is non-finite, or if **all** bands equal the nodata value (`cfg.nodata`, else the file's). Without a nodata value, pixels that are 0 in every band are invalid: the usual fill of mosaics and exports, which would otherwise enter the band statistics. Band names come from the file's band descriptions, falling back to `B<n>`.

## Used by

[[train]] · [[predict]]

## See also

[[band_stats]]
