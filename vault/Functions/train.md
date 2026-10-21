---
tags: [function, pipeline]
module: eoforge.pipeline
---
# train

> Train or fine-tune a model from a config, validate on held-out blocks or images, and map every image.

**Module:** [[eoforge.pipeline]] · **Source:** `eoforge/pipeline.py` · **Kind:** function

```python
train(cfg: Config, config_dir: Path | None = None) -> Path
```

## Parameters

| Name | Type | Meaning |
|---|---|---|
| `cfg` | `Config` | from [[load_config]] |
| `config_dir` | `Path` | folder for relative custom model files |

## Returns

Path to `best.pt`.

## How it works

1. **Data**: `_load_scenes` reads the single image (`data.path` via [[get_source]]) or every entry of `scenes`, with [[read_scene]] and [[rasterise_labels]]; invalid pixels are set to ignore. All images must give the same class list and band count. `_pooled_stats` computes band mean and SD over the valid pixels of all images together; [[normalise]] applies them.
2. **Split**: with `train.val_scenes`, the named images are validation and the rest training (transfer to unseen places). Otherwise [[block_split]] splits each image into spatial blocks ([[Spatial block validation]]). [[ChipDataset]] for train (random, augmented, radiometric jitter) and val (fixed).
3. **Model**: [[build_model]]; optional [[_load_weights]] and encoder freezing.
4. **Optimise**: cross-entropy (optional [[_class_weights]], off by default), AdamW, one-cycle learning rate.
5. **Each epoch**: validate; keep the best epoch by `select_metric` ([[Checkpoint selection]]) as `best.pt`; `last.pt` at the end.
6. **Final check**: reload `best.pt`, map every image with [[predict_scene]], score every held-out pixel with [[confusion]]/[[summarise]], write `metrics.json` and the rasters via [[write_outputs]].

Writes to `runs/<name>/`: `best.pt`, `last.pt`, `metrics.json` (`validation_heldout`, `training_pixels`, `per_scene` for several images, `history`), `prediction/classes.tif` and `entropy.tif` (one subfolder per image when there are several).

## Used by

[[eoforge.cli]] · `tools/run_showcase.py`
