---
tags: [module, data]
---
# eoforge.data.dataset

Spatial train/validation split and the chip sampler that feeds training, from one or several images, with geometric and radiometric augmentation.

**Functions:** [[block_split]] · [[ChipDataset]]

**Private:** `_pad(img, msk, chip, ignore)` pads chips at scene edges to full size (image with 0, mask with ignore).

Concepts: [[Spatial block validation]] · [[Class balancing]]
