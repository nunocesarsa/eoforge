---
tags: [guide, models]
---
# Bring your own model

Any architecture works if it follows the [[Model contract]]: a builder function that returns a module mapping `(B, C, H, W)` to logits `(B, K, H, W)`.

```python
# my_models.py
from torch import nn

class TinyNet(nn.Module):
    def __init__(self, in_channels, num_classes, width=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, width, 3, padding=1), nn.ReLU())
        self.head = nn.Conv2d(width, num_classes, 1)
    def forward(self, x):
        return self.head(self.encoder(x))

def build(in_channels, num_classes, width=32):
    return TinyNet(in_channels, num_classes, width)
```

```yaml
model:
  arch: my_models.py:build      # relative to the config file
  params: {width: 48}
```

Other options for `model.arch`:
- `unet`, `fcn`: [[eoforge.models.builtin]]
- `timm:<encoder>`: [[TimmSeg]]
- `package.module:build`: an installed package
- your own registered name, via [[register_model]]

[[build_model]] runs a test forward pass, so a model that breaks the contract fails immediately with a readable message. Expose `self.encoder` to make `freeze_encoder` work.

Worked example: `examples/my_model.py` ([[Test - Real Sentinel-2 Slovenia]] uses it).
