---
tags: [function, zoo]
module: eoforge.zoo
---
# run_model

> Check and prepare the input, run a model card on a GeoTIFF, and write the results.

**Module:** [[eoforge.zoo]] · **Source:** `eoforge/zoo/run.py` · **Kind:** function

```python
run_model(card | id, image_path, out_dir=None, bands=None, prefix=None, device='auto',
          options=None, adapter=None, log=print, progress=None, windows=None) -> dict
```

## Parameters

| Name | Type | Meaning |
|---|---|---|
| `card` | `ModelCard | str` | card or card id |
| `bands` | `list[int]` | 1-based bands in the card's order; matched by name if omitted ([[match_bands]]) |
| `prefix` | `str` | band name prefix for seasonal stacks, e.g. `dry` |
| `windows` | `list[dict]` | two-date models: `[{image, prefix, bands}, ...]`, one per date |
| `options` | `dict` | model params to override (`tile`, `overlap`, ...), plus `offset`, `match_resolution`, `write_probabilities` |
| `adapter` | `Adapter` | an already loaded adapter (the app reuses them) |

## Returns

The summary dict, also saved as `summary.json`: model, image, bands used, device, run time, warnings, options, class areas, output files, folder.

## Raises

- `BandError`: the image lacks bands the model needs (lists them)
- `ImportError`: a package is missing; the message gives the install command

## How it works

1. **Read** (`read_input`): bands by name, with the card's `band_fallback` stand-ins; images without band names via [[guess_descriptions]]; a seasonal stack without a prefix uses its first band set, with a warning. Two-date models read one band set per date; a second image on another grid is warped onto the first (average when coarsening, bilinear otherwise). With `match_resolution` the image is resampled to the card's `resolution_m` first ([[grid_at_resolution]]).
2. **Valid pixels**: finite, and not equal to nodata in every band; without a nodata value, pixels that are 0 in every band are no data.
3. **Units** ([[check_units]]): 0-1 reflectance is scaled to x10000, `offset` is subtracted, and a warning is given when the data look like they carry the Sentinel-2 +1000 offset. Cards with `params.input: any` are left alone.
4. **Run** the [[Adapter]]; its own warnings (e.g. band shift for trained models, input scale for TerraTorch models) are added to the summary.
5. **Write** `<out_dir>/<image>__<model>/`: `classes.tif` (or `objects.tif`), `entropy.tif` when the model gives probabilities, optional `probabilities.tif`, extra rasters, `fields.gpkg`/`objects.gpkg` for vector models, `summary.json` and `preview_*.png`.

Also warns when the pixel size differs from the card's `resolution_m` by more than 25%, and when the image is in geographic coordinates (areas then approximate).

## Used by

[[eoforge.cli]] · [[eoforge.app]]

## See also

[[Model card]] · [[Adapter]] · [[Sample datasets per model]]
