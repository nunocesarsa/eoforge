---
tags: [test, synthetic]
date: 2026-10-19
---
# Test: synthetic demo

> Checks the mechanics on data where the full truth is known.

![[demo_result.png]]

**Data:** [[make_demo]]: 512 × 512 px, 6 bands (B2 B3 B4 B8 B11 B12), 6 classes, EPSG:32734 near the Zambezi region; 270 label features (120 small polygons, 150 points buffered 10 m); a second scene of a different landscape for transfer.

| Model | Parameters | Held-out OA | Kappa | Macro-F1 |
|---|---|---|---|---|
| `unet`, 15 epochs | 1.09 M | 0.991 | 0.989 | 0.988 |
| custom `examples/my_model.py`, 6 epochs | 0.03 M | 0.990 | 0.988 | 0.983 |
| `timm:resnet18`, 6 epochs | 11.3 M | 0.985 | 0.981 | 0.978 |
| `unet` fine-tuned, encoder frozen, 3 epochs | 0.42 M trainable | 0.989 | 0.985 | 0.984 |

(U-Net row rerun after the [[Changelog]] changes; the others are from the first run.)

- Wall-to-wall agreement with the full truth: **0.973** on the training scene, **0.953** on the unseen scene.
- Mean entropy **0.16** on correct pixels vs **0.63** on wrong ones ([[Entropy as uncertainty]]); errors sit on class boundaries and in grassland vs shrubland.
- ONNX export matches PyTorch to ~2e-6 ([[export_onnx]]).

The task is easy by design; this confirms the pipeline is wired correctly, not how a model performs on real imagery. For that, see [[Test - Real Sentinel-2 Slovenia]].
