---
tags: [project]
---
# Changelog

## 0.6.0 (2026-10-21), generalisation review and first public release
Findings and reasons are in [[Review - generalisation 0.6.0]]; test data per model in [[Sample datasets per model]].
- **Fixed** Prithvi burn scars got inputs 10000x too large: TerraTorch's generic datamodule reports `constant_scale: 1` by default, which overrode the card's `scale: 0.0001`. The card now wins, and TerraTorch models warn when the normalised input is far from the training data.
- **Fixed** TerraMind maps were not repeatable: TerraMind draws its sampling seed from Python's `random`; the seed is now set there too.
- **Fixed** Delineate Anything failed on images smaller than a tile whose size was not a multiple of 32.
- **Fixed** pixels that are 0 in every band counted as data when the file has no nodata value (zoo and training): they skewed band statistics and class areas.
- **Fixed** the app kept switching back to the uploaded image while the file stayed in the uploader; a new upload with an existing name no longer reuses the old file.
- **Fixed** training over-corrected for rare classes by default: `train.class_weights` is now off (chip sampling is already class-balanced).
- **Added** input checks in [[run_model]]: 0-1 reflectance is scaled to x10000; `offset` (CLI `--offset 1000`, app *Subtract offset*) removes the Sentinel-2 +1000 offset of processing baseline 04.00+; a warning when data look like they carry it.
- **Added** `match_resolution` (CLI `--match-resolution`, app checkbox): resample to the model's pixel size, e.g. 30 m for the HLS burn scar model.
- **Added** images without band names: 13 bands are read as Sentinel-2 L1C order, 12 as L2A order, and an image with exactly the model's number of bands as the model's order (FTW chips).
- **Added** `B8A -> B08` stand-ins for OmniCloudMask and both Prithvi models; a warning naming the band set used when a seasonal stack is read without a prefix.
- **Added** multi-image training: `scenes: [{name, image, labels}, ...]`, statistics pooled over all images, and `train.val_scenes` to validate on whole images the model never saw. `run_showcase.py --pooled --holdout 3` trains one model on all LandCoverNet chips.
- **Added** `train.augment_radiometric` (default 0.1): per-chip band gain/offset jitter so models depend less on one image's brightness.
- **Added** `tools/get_tutorial_data.py` (Sentinel-2 L2A scenes, ESA WorldCover and Sen1Floods11 chips for one area, plus TUTORIAL.md) and `tools/compare_maps.py` (cross-table of a result against a reference map).
- **Added** repository files for GitHub: `LICENSE` (MIT), `THIRD_PARTY_NOTICES.md`, `.gitignore` (data, runs and results stay local; the example U-Net is kept); `install.sh` now defaults to `./.venv` and has no machine-specific paths; new README.
- **Removed** the vault generator scripts (`tools/make_vault.py`, `vault_extra.py`, `vault_zoo.py`): the vault is now edited directly.
- **Changed** tutorial: land cover runs on the green-season scene first ([[Test - Tutorial Zambia]]); `--text-only`.
- **Added** band-shift warnings when a trained model runs on a new image in the app; `eoforge predict` picks bands by name when the new image orders them differently.

## 0.5.0 (2026-10-21), models for land cover and fields in Africa
- **Added** TerraMind generated land cover (`terramind-lulc`), Fields of The World (`ftw-fields`, `ftw-fields-fast`, `ftw-fields-1date`) and Delineate Anything (`delineate-anything`, `delineate-anything-small`) ([[Model zoo and app]]).
- **Added** two-date inputs (`params.windows`, `run_model(windows=...)`, *Start/End of season* in the app) and band stand-ins (`params.band_fallback`).
- **Added** field polygons (fields.gpkg) for FTW and Delineate Anything.
- **Changed** model list order in the app: land cover, fields, then the rest.

## 0.4.1 (2026-10-20)
- **Changed** every model library is now a normal dependency; `install.sh` installs PyTorch for the GTX 1070, EOForge, all model libraries and SAM 2, and downloads all weights ([[Model zoo and app]]).
- **Added** `eoforge zoo download` and `eoforge zoo status`.
- **Fixed** SAM 2 needed the `sam2` package, which `segment-geospatial` does not install by default.

## 0.4.0 (2026-10-20), model zoo and web app
- **Added** the model zoo: [[Model card]]s, [[Adapter]]s and [[run_model]]. Ready-made models: Dynamic World (converted to ONNX, included), Prithvi-EO-2.0 floods and burn scars (TerraTorch), OmniCloudMask, SAM 2 (segment-geospatial), and every eoforge model you trained.
- **Added** `eoforge app`: a local web app to pick an image and a model, run it on the GPU and see the result on a map, plus an *Add a model* form ([[Model zoo and app]]).
- **Added** `eoforge zoo list | info | run`.
- **Added** install groups `zoo`, `prithvi`, `clouds`, `sam`, and `requirements/pascal-gpu-constraints.txt` to keep PyTorch 2.14 + CUDA 12.6 for the GTX 1070.

## 0.3.1 (2026-10-20), after showcase run 1
- **Fixed** `fetch_landcovernet.py` date choice: picked the first clear dates of the year (often only January–March); now one composite per season (wet/dry), dates spread through each season, stacked as 20 bands ([[Test - LandCoverNet showcase run 1]]).
- **Fixed** `fetch_landcovernet.py` found no chips on the real bucket: files are now found by listing and matching names, skipped chips print a reason, and `--inspect` shows the layout.
- **Changed** `run_showcase.py`: class-balanced sampling only by default (no extra loss weights); `--class-weights`; gallery uses the dry-season image; clear message when no chips were downloaded.

## 0.3.0 (2026-10-20)
- **Added** raster labels: `labels.path` can be a GeoTIFF; `labels.class_map` and `labels.band` ([[rasterise_labels]]).
- **Added** `tools/fetch_landcovernet.py` and `tools/run_showcase.py` ([[LandCoverNet showcase]]); optional dependency group `showcase`.

## 0.2.0 (2026-10-19), after the real-data test
- **Added** `labels.exclude` to drop label values ([[rasterise_labels]]).
- **Added** [[check_band_shift]]: [[predict]] warns when a band's mean is > 2 SD from the training data ([[Band shift]]).
- **Changed** best-epoch selection to validation loss; `train.select_metric` allows `macro_f1`, `overall_accuracy`, `kappa` ([[Checkpoint selection]]).
- **Added** validation loss to the training log and history.
- **Fixed** custom model files given relative to the config are stored with an absolute path in the checkpoint, so `predict` works from any folder.
- **Added** `examples/real_s2_slovenia.py`, `configs/real_s2_*.yaml`, `tools/real_test_report.py` ([[Test - Real Sentinel-2 Slovenia]]).

## 0.1.0 (2026-10-19)
- First pipeline: local GeoTIFF + vector labels → fine-tune → class map + entropy, ONNX export ([[Test - Synthetic demo]]).
