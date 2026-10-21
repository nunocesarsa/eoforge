---
tags: [test, review]
---
# Review: generalisation (0.6.0)

An adversarial read of the training pipeline, the model zoo and the app, looking for bugs and for reasons models do not transfer to new images. Fixes are in [[Changelog]] 0.6.0; data to test each model with are in [[Sample datasets per model]].

## Starting point: Dynamic World on LandCoverNet 35KNT_09
Result: shrub 57%, bare 32%, grass 7%, crops 3%, trees 0%, against labels that are about half woody.

| Label (LandCoverNet) | Dynamic World said |
|---|---|
| woody (34,984 px) | shrub 74%, grass 12%, bare 10%, trees 3% |
| semi-natural vegetation | bare 54%, shrub 42% |
| cultivated | bare 56%, shrub 38% |
| artificial bare ground | bare 69%, shrub 24% |

Normalisation, band order and colours were checked against Google's reference code and are correct. The map is spatially coherent. The gap comes from the input and the class definitions:
1. the image is an L2A composite and Dynamic World was trained on L1C;
2. open savanna woodland is *shrub and scrub* in Dynamic World's scheme;
3. a single wet-season composite.

## Bugs found and fixed
| Where | Problem | Effect |
|---|---|---|
| Prithvi burn scars | TerraTorch's generic datamodule reports `constant_scale: 1` by default, which replaced the card's `scale: 0.0001` | inputs 10000x too large; the model could not work |
| TerraMind | seed set in torch only; TerraMind seeds its sampler from Python `random` | a different map on every run |
| Delineate Anything | small images: tile x 2 not a multiple of 32 | YOLO rejects the tile |
| zoo and training | all-zero fill pixels counted as data when the file has no nodata | skewed statistics, areas and predictions near edges |
| app | uploaded image kept re-selecting itself; same-name uploads reused the old file | could not switch images after an upload |
| training | loss weights plus class-balanced sampling by default | rare classes over-predicted |
| CLI predict | bands taken by position from training | silent nonsense if the new image orders bands differently |

## Generalisation problems and what was added
| Problem | Added |
|---|---|
| Data in 0-1 reflectance (common in exports) given to models expecting x10000 | automatic scaling with a warning |
| Sentinel-2 since 2022 carries +1000 | `offset` option and a warning when every band is above 1000 |
| Wrong pixel size (10 m S2 into the 30 m HLS model) | `match_resolution` |
| Unnamed bands (Sen1Floods11, FTW chips) could not be matched | Sentinel-2 order for 13/12 bands, model order when the count matches |
| Missing B8A | `B8A -> B08` stand-ins for OmniCloudMask and Prithvi |
| One-chip training: one place, one date, statistics of one image | `scenes` (several images, pooled statistics), `val_scenes` (whole unseen images), `run_showcase.py --pooled` |
| Model learns the brightness of one image | `train.augment_radiometric` |
| Trained model on a different image without warning | band-shift warning in the zoo/app |

## Afterwards on your GPU
- [[Test - LandCoverNet pooled run]]: one model for 7 chips, kappa 0.23 on 3 unseen chips; it overfits after 3 epochs.
- [[Test - Tutorial Zambia]]: tutorial data download worked; Dynamic World's 78 % bare in late August is the dry season, not a bug.

## Still open
- No L2A-to-L1C conversion: Dynamic World and Prithvi floods need L1C input.
- Two-date FTW needs real start/end-of-season images; LandCoverNet wet/dry composites are an approximation.
- TerraMind tiles are generated independently; seams can show at tile edges.

## Tests (small, synthetic, in the development copy)
Multi-scene training with a held-out scene; single-scene training and `eoforge predict`; 0-1 float input to Dynamic World (scaled, nodata corner masked); offset hint and subtraction; 10 m to 30 m resampling; two-date reading from two files at different resolutions; unnamed 13-band and 4-band images; the app run end to end (Streamlit AppTest).
