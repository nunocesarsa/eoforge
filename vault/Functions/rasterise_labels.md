---
tags: [function, raster]
module: eoforge.data.raster
---
# rasterise_labels

> Put vector or raster labels onto the image grid.

**Module:** [[eoforge.data.raster]] · **Source:** `eoforge/data/raster.py` · **Kind:** function

```python
rasterise_labels(cfg: LabelConfig, profile: dict) -> (mask, classes)
```

## Parameters

| Name | Type | Meaning |
|---|---|---|
| `cfg` | `LabelConfig` | path, class field, classes, exclude, point_buffer, class_map, band, ignore_index |
| `profile` | `dict` | rasterio profile of the image |

## Returns

`mask` (H, W uint8, class ids, `ignore_index` elsewhere) and `classes` (names in id order).

## Raises

- `ValueError`: missing class field, no CRS, no labels left after `exclude`/mapping, or values not in `labels.classes`

## How it works

**Raster labels** (`.tif`, `.tiff`, `.vrt`, `.img`, `.jp2`) go to `_raster_labels`: the label band is resampled onto the image grid with nearest neighbour (any CRS or resolution), codes are mapped to names with `class_map` (or each code becomes its own class), and nodata, unmapped codes and `exclude` names become `ignore_index`. Verified identical to the vector route on the Slovenia data, and 98.6% after a round trip through EPSG:4326.

**Vector labels:**
1. Read with geopandas; drop empty geometries, empty class values and `exclude` values.
2. Reproject to the image CRS.
3. Class ids follow `labels.classes`, or sorted unique values.
4. Points are buffered by `point_buffer` metres.
5. Rasterise with `all_touched=True` so small polygons and points hit at least one pixel. Later features overwrite earlier ones where they overlap.

## Used by

[[train]]
