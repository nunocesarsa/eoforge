---
tags: [function, predict]
module: eoforge.predict
---
# predict_scene

> Tiled, blended inference over a whole normalised scene.

**Module:** [[eoforge.predict]] · **Source:** `eoforge/predict.py` · **Kind:** function

```python
predict_scene(model, image, valid, num_classes, tile=256, overlap=32, batch_size=8, device='cpu') -> np.ndarray
```

## Returns

Class probabilities (K, H, W) float32.

## How it works

See [[Tiled inference]]. The tile size is capped at the scene size; edge tiles are zero-padded and cropped back.

## Used by

[[train]] · [[predict]]
