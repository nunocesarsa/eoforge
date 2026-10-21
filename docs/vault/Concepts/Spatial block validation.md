---
tags: [concept, validation]
---
# Spatial block validation

Nearby pixels in EO imagery are highly correlated. If validation pixels are randomly mixed with training pixels, the model is tested on near-copies of what it trained on and accuracy is overstated.

[[block_split]] cuts the scene into square blocks of `train.block_size` pixels, keeps only blocks containing labels, and holds out a random `val_fraction` of them. Every pixel in a held-out block is used only for validation.

After training, [[train]] maps the whole scene with the best model and scores **every** held-out pixel, not just sampled chips. That number goes to `metrics.json` as `validation_heldout_blocks`.

Choosing `block_size`:
- larger than the typical patch/field size, so a field does not straddle train and val too often;
- small enough that there are at least ~10 labelled blocks.

Caveat: rare classes can end up mostly in one split. Check `n_ref` per class in `metrics.json`.
