---
tags: [function, metrics]
module: eoforge.metrics
---
# confusion

> Confusion matrix (rows = reference, columns = predicted), ignoring unlabelled pixels.

**Module:** [[eoforge.metrics]] · **Source:** `eoforge/metrics.py` · **Kind:** function

```python
confusion(y_true, y_pred, k, ignore) -> np.ndarray
```

## Used by

[[train]] · [[summarise]]
