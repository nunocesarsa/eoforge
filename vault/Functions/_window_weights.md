---
tags: [function, predict]
module: eoforge.predict
---
# _window_weights

> 2-D blending weights: 1 in the middle, ramping linearly to 0.05 over `overlap` pixels at each edge.

**Module:** [[eoforge.predict]] · **Source:** `eoforge/predict.py` · **Kind:** private function

```python
_window_weights(size, overlap) -> np.ndarray
```

## Used by

[[predict_scene]]

## See also

[[Tiled inference]]
