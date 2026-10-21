---
tags: [function, timm_seg]
module: eoforge.models.timm_seg
---
# TimmSeg

> Any timm CNN encoder (`features_only`) with a light FPN-style decoder.

**Module:** [[eoforge.models.timm_seg]] · **Source:** `eoforge/models/timm_seg.py` · **Kind:** class (nn.Module)

```python
TimmSeg(encoder: str, in_channels, num_classes, pretrained=False, dim=64)
```

## Parameters

| Name | Type | Meaning |
|---|---|---|
| `encoder` | `str` | timm model name, e.g. `resnet18`, `efficientnet_b0`, `convnext_tiny` |
| `pretrained` | `bool` | download ImageNet weights; timm adapts the first conv to the band count |
| `dim` | `int` | decoder channels |

## How it works

Each feature level goes through a 1×1 lateral conv; levels are summed top-down with bilinear upsampling, fused by a 3×3 conv, and the logits are upsampled to the input size. Use as `arch: timm:<encoder>`.

## Used by

[[build_model]]
