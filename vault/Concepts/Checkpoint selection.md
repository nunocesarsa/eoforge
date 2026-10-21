---
tags: [concept, training, lesson]
---
# Checkpoint selection

After every epoch, [[train]] evaluates the validation chips and keeps the best epoch as `best.pt`. Which epoch counts as best is set by `train.select_metric`:

- `val_loss` (default): weighted cross-entropy on validation chips. Smooth, uses every pixel's probability.
- `macro_f1`, `overall_accuracy`, `kappa`: available, but noisy when some classes have few validation pixels.

Why the default changed: in [[Test - Real Sentinel-2 Slovenia]], selecting by macro-F1 picked epoch 2 of the U-Net, because one class had only 14 validation pixels and its F1 jumped around. That model was barely trained (entropy 0.94 everywhere). See [[Changelog]].
