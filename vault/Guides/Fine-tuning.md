---
tags: [guide, models]
---
# Fine-tuning

```yaml
model:
  arch: unet
  params: {base: 16, depth: 2}
  weights: ../runs/kaza_2020/best.pt
  freeze_encoder: true
train:
  lr: 0.001
  epochs: 10
```

- [[_load_weights]] loads every tensor whose name and shape match, and skips the rest. A new head for a different number of classes is therefore trained from scratch while the encoder is reused.
- With `freeze_encoder: true`, [[encoder_parameters]] is frozen and only the decoder and head train.
- Use the same bands (in the same order) as the source model; normalisation stats are recomputed for the new scene.

In [[Test - Real Sentinel-2 Slovenia]], a U-Net pretrained on 2016 and fine-tuned on 2017 trained with 38% of its parameters.
