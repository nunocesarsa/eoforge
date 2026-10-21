---
tags: [module, models]
---
# eoforge.models.builtin

Small architectures that accept any number of input bands.

- [[UNet]]: encoder-decoder with skip connections
- [[FCN]]: plain stack of 3×3 convolutions at full resolution

**Private:** `_block(cin, cout, dropout)`: two conv-BN-ReLU layers, optional `Dropout2d`.
