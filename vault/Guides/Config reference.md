---
tags: [guide, config]
---
# Config reference

Loaded by [[load_config]]. Unknown keys raise an error, so typos fail loudly. Relative paths are resolved from the config file's folder.

## Top level
| Key | Default | Meaning |
|---|---|---|
| `name` | `run` | Run name; outputs go to `output_dir/name` |
| `output_dir` | `runs` | Parent folder for runs |
| `scenes` | `[]` | Several images, each with its own labels: `[{name: a, image: a.tif, labels: a_labels.tif}, ...]`. Replaces `data.path` and `labels.path`; the other `data.*` and `labels.*` settings apply to all. Band statistics are pooled over all images |

## `data` (DataConfig)
| Key | Default | Meaning |
|---|---|---|
| `source` | `local` | `local`, `stac`, `gee` ([[eoforge.data.sources]]; only `local` works today) |
| `path` | | GeoTIFF for `source: local` |
| `bands` | all | 1-based band indices, e.g. `[2,3,4,8]` |
| `band_names` | from file | Override band names |
| `nodata` | from file | Override nodata value |
| `params` | `{}` | Source-specific options (for STAC/GEE later) |

## `labels` (LabelConfig)
| Key | Default | Meaning |
|---|---|---|
| `path` | | Vector file (gpkg, shp, geojson...) **or** label raster (GeoTIFF), in any CRS |
| `class_field` | `class` | Attribute holding the class |
| `classes` | sorted unique | Class order = class ids. Recommended so ids are stable across runs |
| `exclude` | `[]` | Label values to drop, e.g. `""`, `"no data"`, very rare classes |
| `point_buffer` | `0` | Metres; turns points into small polygons (vector labels) |
| `class_map` | none | Raster labels: `{code: class name}`, e.g. `{1: water, 5: woody}`; codes not listed are ignored |
| `band` | `1` | Raster labels: band holding the class codes |
| `ignore_index` | `255` | Value for unlabelled pixels |

## `model` (ModelConfig)
| Key | Default | Meaning |
|---|---|---|
| `arch` | `unet` | See [[build_model]] and [[Bring your own model]] |
| `params` | `{}` | Passed to the builder |
| `weights` | none | Checkpoint to start from ([[Fine-tuning]]) |
| `freeze_encoder` | `false` | Freeze `model.encoder` ([[encoder_parameters]]) |

## `train` (TrainConfig)
| Key | Default | Meaning |
|---|---|---|
| `chip_size` | 64 | Training chip size in pixels |
| `chips_per_epoch` | 512 | Chips sampled per epoch |
| `batch_size` | 16 | |
| `epochs` | 20 | |
| `lr` | 0.001 | Peak learning rate (one-cycle schedule) |
| `weight_decay` | 0.0001 | AdamW |
| `val_fraction` | 0.2 | Fraction of labelled blocks held out |
| `block_size` | 256 | Block size in pixels for [[Spatial block validation]] |
| `class_weights` | `null` | `null`, `balanced` or a list ([[Class balancing]]); chips are already class-balanced |
| `augment_radiometric` | 0.1 | Per-chip band gain/offset jitter in normalised units (0 = off); helps transfer to other images and dates |
| `val_scenes` | none | With `scenes`: names of whole images held out for validation (the honest transfer test). Without it, each image is split into spatial blocks |
| `augment` | `true` | Flips and 90° rotations |
| `select_metric` | `val_loss` | Which epoch to keep ([[Checkpoint selection]]) |
| `seed` | 42 | Seeds sampling, split and initialisation |
| `device` | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `num_threads` | none | CPU threads for torch |

## `predict` (PredictConfig)
| Key | Default | Meaning |
|---|---|---|
| `tile_size` | 256 | Inference tile size ([[Tiled inference]]) |
| `overlap` | 32 | Overlap per tile edge |
| `batch_size` | 8 | Tiles per batch |
| `write_entropy` | `true` | Write `entropy.tif` ([[Entropy as uncertainty]]) |
| `write_probabilities` | `false` | Write `probabilities.tif` |
