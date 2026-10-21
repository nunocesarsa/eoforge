---
tags: [function, pipeline]
module: eoforge.pipeline
---
# _class_weights

> Loss weights per class.

**Module:** [[eoforge.pipeline]] · **Source:** `eoforge/pipeline.py` · **Kind:** private function

```python
_class_weights(mask, k, ignore, setting) -> Tensor | None
```

## How it works

`balanced`: `total / (k * count)` per class, rescaled to mean 1 over present classes (absent classes get 0). A list is used as given. `None` disables weighting.

## Used by

[[train]]

## See also

[[Class balancing]]
