---
tags: [function, predict]
module: eoforge.predict
---
# write_outputs

> Write the class map and optional uncertainty and probability rasters.

**Module:** [[eoforge.predict]] · **Source:** `eoforge/predict.py` · **Kind:** function

```python
write_outputs(out_dir, profile, probs, valid, classes, write_entropy=True, write_probabilities=False) -> dict
```

## How it works

| File | Type | Content |
|---|---|---|
| `classes.tif` | uint8 | argmax class, 255 = nodata, colour table, class names in tags `class_0`... |
| `entropy.tif` | float32 | normalised entropy, NaN = nodata |
| `probabilities.tif` | float32 × K | one band per class, named |

All are deflate-compressed, tiled, with the source CRS and transform.

## Used by

[[train]] · [[predict]]
