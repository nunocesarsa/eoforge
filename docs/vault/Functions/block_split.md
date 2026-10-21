---
tags: [function, dataset]
module: eoforge.data.dataset
---
# block_split

> Split labelled pixels into train and validation by spatial blocks.

**Module:** [[eoforge.data.dataset]] · **Source:** `eoforge/data/dataset.py` · **Kind:** function

```python
block_split(mask, ignore, block, val_fraction, seed) -> (train_mask, val_mask)
```

## Parameters

| Name | Type | Meaning |
|---|---|---|
| `mask` | `ndarray` | label mask |
| `ignore` | `int` | unlabelled value |
| `block` | `int` | block size in pixels |
| `val_fraction` | `float` | fraction of labelled blocks held out (at least one) |
| `seed` | `int` | shuffle seed |

## Returns

Two copies of the mask, each with the other split set to `ignore`.

## Raises

- `ValueError`: labels fall in fewer than 2 blocks

## Used by

[[train]]

## See also

[[Spatial block validation]]
