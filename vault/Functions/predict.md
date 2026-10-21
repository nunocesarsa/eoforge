---
tags: [function, pipeline]
module: eoforge.pipeline
---
# predict

> Apply a trained checkpoint to any GeoTIFF with the same bands.

**Module:** [[eoforge.pipeline]] · **Source:** `eoforge/pipeline.py` · **Kind:** function

```python
predict(checkpoint, image_path, out_dir, cfg=None, device='auto', base_dir=None) -> dict
```

## Parameters

| Name | Type | Meaning |
|---|---|---|
| `checkpoint` | `path` | `best.pt` |
| `image_path` | `path` | GeoTIFF |
| `out_dir` | `path` | output folder |
| `cfg` | `Config` | optional, only `predict` settings are used |

## Returns

Dict of written file paths.

## Raises

- `ValueError`: band count differs from the model

## How it works

Selects bands by name when the image has the training band names (so a different band order still works), otherwise by the checkpoint's band numbers; uses the training normalisation stats, runs [[check_band_shift]], then [[predict_scene]] and [[write_outputs]].

## Used by

[[eoforge.cli]]

## See also

[[Band shift]]
