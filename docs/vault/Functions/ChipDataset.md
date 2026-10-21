---
tags: [function, dataset]
module: eoforge.data.dataset
---
# ChipDataset

> PyTorch dataset of chips centred near labelled pixels, from one or several images.

**Module:** [[eoforge.data.dataset]] · **Source:** `eoforge/data/dataset.py` · **Kind:** class (torch Dataset)

```python
ChipDataset(image, mask, chip, n, ignore, augment=False, seed=0, fixed=False, radiometric=0.0)
```

## Parameters

| Name | Type | Meaning |
|---|---|---|
| `image` | `ndarray` or list | normalised (C, H, W), or one per image for multi-image training |
| `mask` | `ndarray` or list | split mask(s), same order as `image` |
| `chip` | `int` | chip size |
| `n` | `int` | chips per epoch |
| `augment` | `bool` | random flips and 90° rotations |
| `fixed` | `bool` | fixed chip centres without jitter (used for validation, so it is comparable across epochs) |
| `radiometric` | `float` | with `augment`: per-chip band gain and offset jitter of this size, in normalised units (shared part + half as much per band) |

## Returns

Items are `(image_chip float32 [C, chip, chip], mask_chip int64 [chip, chip])`.

## How it works

Labelled pixels of all images are pooled; centres are drawn with class-balanced probabilities over the pool ([[Class balancing]]), so a class that is rare in every image still gets its share. Centres are jittered by up to a quarter chip so labelled pixels are not always central. Windows are clamped inside the image; chips at edges are padded with ignore.

The radiometric jitter (`train.augment_radiometric`, default 0.1) imitates the brightness and contrast differences between dates, atmospheres and processing levels, so the model leans less on the exact values of the training images ([[Normalisation]]).

## Used by

[[train]]
