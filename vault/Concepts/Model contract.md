---
tags: [concept, models]
---
# Model contract

The single rule an architecture must follow to be ingested by EOForge:

```python
build(in_channels: int, num_classes: int, **params) -> torch.nn.Module
model(x: Tensor[B, C, H, W]) -> Tensor[B, num_classes, H, W]   # logits, same H and W
```

- `in_channels` is the number of selected bands; `num_classes` the number of classes in the labels.
- Output must be raw logits (no softmax); softmax is applied in [[predict_scene]] and the loss is cross-entropy.
- Should accept H and W that are not powers of two (the built-in [[UNet]] pads internally).
- Optional: an `encoder` attribute (an `nn.Module`) so `freeze_encoder` can freeze it ([[encoder_parameters]]).

Checked by [[_check_contract]] with a `(1, C, 64, 64)` input when the model is built.
