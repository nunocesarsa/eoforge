---
tags: [function, builtin]
module: eoforge.models.builtin
---
# UNet

> U-Net with `depth` down/up levels and skip connections.

**Module:** [[eoforge.models.builtin]] · **Source:** `eoforge/models/builtin.py` · **Kind:** class (nn.Module)

```python
UNet(in_channels, num_classes, base=32, depth=3, dropout=0.1)
```

## Parameters

| Name | Type | Meaning |
|---|---|---|
| `base` | `int` | channels at the first level; doubles per level |
| `depth` | `int` | number of pooling steps |
| `dropout` | `float` | Dropout2d in the bottleneck |

## How it works

Inputs are padded to a multiple of `2**depth` and cropped back, so any tile size works. `encoder` holds the down path and bottleneck (for freezing).

## Used by

[[build_model]]
