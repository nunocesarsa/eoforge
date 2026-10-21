---
tags: [function, metrics]
module: eoforge.metrics
---
# summarise

> Accuracy metrics from a confusion matrix.

**Module:** [[eoforge.metrics]] · **Source:** `eoforge/metrics.py` · **Kind:** function

```python
summarise(cm, classes) -> dict
```

## Returns

`n`, `overall_accuracy`, `kappa` (Cohen's), `macro_f1` (over classes with values), `per_class` with `producers_acc` (recall), `users_acc` (precision), `f1`, `n_ref`, and `confusion_matrix`.

## Used by

[[train]]
