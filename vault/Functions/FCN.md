---
tags: [function, builtin]
module: eoforge.models.builtin
---
# FCN

> Stack of 3×3 conv-BN-ReLU layers at full resolution with a 1×1 head.

**Module:** [[eoforge.models.builtin]] · **Source:** `eoforge/models/builtin.py` · **Kind:** class (nn.Module)

```python
FCN(in_channels, num_classes, width=64, layers=4)
```

## How it works

No downsampling, so a small receptive field (2×layers+1 pixels). Fast and a useful baseline; in [[Test - Real Sentinel-2 Slovenia]] it was competitive with much larger models.

## Used by

[[build_model]]
