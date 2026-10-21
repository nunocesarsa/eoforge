---
tags: [concept, data]
---
# Normalisation

## Your trained models
Each band is standardised as `(x - mean) / std`, using the mean and standard deviation of valid pixels in the **training** image(s) ([[band_stats]], [[normalise]]). With several images (`scenes`) the statistics are pooled over all of them. Nodata pixels are set to 0 after normalisation.

The stats are stored in the checkpoint ([[Checkpoint format]]) and reused at prediction time, so a new image is normalised with the training statistics. If its distribution differs a lot, the model sees out-of-range inputs: see [[Band shift]]. Training with several images and with radiometric jitter (`train.augment_radiometric`, [[ChipDataset]]) makes models less sensitive to that.

## Zoo models
Each published model has its own normalisation, set in its [[Model card]] (`params.normalise`, `params.scale`) or taken from the model's own config (TerraTorch). All of them assume Sentinel-2 style reflectance x 10000 as input; [[check_units]] converts 0-1 reflectance and removes the 2022 offset, and the TerraTorch adapter warns when the normalised input sits more than 8 SD from the training data, the sign of a units mistake.
