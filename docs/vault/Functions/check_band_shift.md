---
tags: [function, pipeline]
module: eoforge.pipeline
---
# check_band_shift

> Warn when a new image's band means are far from the training statistics.

**Module:** [[eoforge.pipeline]] · **Source:** `eoforge/pipeline.py` · **Kind:** function

```python
check_band_shift(scene, mean, std, band_names, threshold=2.0) -> dict
```

## Returns

Dict of band → shift in training SDs. Logs a warning for bands beyond ±`threshold`.

## Used by

[[predict]]

## See also

[[Band shift]]
