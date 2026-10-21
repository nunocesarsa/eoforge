---
tags: [function, zoo]
module: eoforge.zoo
---
# tiled_probs

> Run a model function over an image in blended, overlapping tiles.

**Module:** [[eoforge.zoo]] · **Source:** `eoforge/zoo.py` · **Kind:** function

```python
tiled_probs(fn, image, valid, k, tile=512, overlap=32, batch=4, pad_mode='reflect', multiple=1, progress=None, fixed=False)
```

## How it works

`fn` maps a batch `(B, C, t, t)` to probabilities `(B, K, t, t)`. Same blending as [[Tiled inference]]. `multiple` pads tiles to a multiple of a size (16 for ViTs); `fixed` keeps the tile size for models with a fixed input size.

## Used by

[[Adapter]]
