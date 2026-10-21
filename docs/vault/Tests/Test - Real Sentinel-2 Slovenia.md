---
tags: [test, real-data]
date: 2026-10-19
---
# Test: real Sentinel-2, Slovenia

> End-to-end run on real Sentinel-2 imagery with official land-use reference polygons: data preparation, five architectures, three spatial splits, transfer to another year, fine-tuning and ONNX export.

![[real_s2_maps.png]]

## Data
- **Imagery:** Sentinel-2 L1C, 13 bands, 68 dates (Jul 2015 – Dec 2017), 10 m, 100 × 101 px (~1 km²) in Slovenia, EPSG:32633. From the `eo-learn` example patch (github.com/sentinel-hub/eo-learn, `example_data/TestEOPatch`).
- **Composites:** cloud-masked median of April–October, built by `examples/real_s2_slovenia.py`. 2017: 18 dates (15–18 clear observations per pixel). 2016: 8 dates (6–8).
- **Labels:** 88 polygons from the Slovenian land-use register (mapped Dec 2017 – Feb 2018). Classes used: forest, grassland, shrubland, artificial surface. Dropped with `labels.exclude`: empty (no-data) polygons and *cultivated land* (11 px).
- **Georeferencing check:** rasterising the polygons with EOForge reproduces eo-learn's own reference raster exactly (100% of pixels).
- **Bands used:** 10 of 13; B01, B09 and B10 removed (see [[Band shift]]).

Imbalance is severe: about 77% forest, 17% grassland, 4% shrubland, 2% artificial surface.

## Setup
- Configs: `configs/real_s2_unet.yaml` and variants; three seeds in `configs/seeds/`.
- Each seed gives a **different spatial split** (25 × 25 px blocks, 25% held out) and a different initialisation, so the three runs act as a small repeated spatial cross-validation ([[Spatial block validation]]).
- 30 epochs (10 for fine-tuning), 48 px chips, AdamW one-cycle, balanced classes ([[Class balancing]]), best epoch by validation loss ([[Checkpoint selection]]).
- **Transfer test:** each 2017 model is applied to the 2016 composite with [[predict]] and scored on the same held-out blocks (land use assumed stable between the years).
- **Baseline:** scikit-learn Random Forest (300 trees, balanced) on per-pixel bands, same splits.

| Seed | Train / val pixels: forest | grassland | shrubland | artificial |
|---|---|---|---|---|
| 1 | 6497 / 1328 | 1198 / 426 | 304 / 111 | 106 / 50 |
| 2 | 6061 / 1764 | 1049 / 575 | 337 / 78 | 56 / 100 |
| 3 | 6023 / 1802 | 1537 / 87 | 408 / 7 | 137 / 19 |

## Results (mean ± SD over 3 spatial splits)

| Model | OA 2017 | Kappa 2017 | Macro-F1 2017 | OA 2016 | Kappa 2016 | Entropy correct / wrong |
|---|---|---|---|---|---|---|
| U-Net | 0.77 ± 0.10 | 0.48 ± 0.07 | 0.51 ± 0.05 | 0.76 ± 0.17 | 0.49 ± 0.16 | 0.76 / 0.76 |
| FCN | 0.71 ± 0.11 | 0.40 ± 0.03 | 0.47 ± 0.02 | 0.69 ± 0.19 | 0.41 ± 0.11 | 0.57 / 0.68 |
| Custom (my_model.py) | 0.77 ± 0.11 | 0.50 ± 0.02 | 0.53 ± 0.01 | 0.79 ± 0.11 | 0.51 ± 0.09 | 0.46 / 0.63 |
| timm ResNet-18 | 0.81 ± 0.08 | 0.51 ± 0.05 | 0.50 ± 0.06 | 0.84 ± 0.08 | 0.56 ± 0.02 | 0.37 / 0.60 |
| U-Net 2016 → fine-tuned 2017 | 0.80 ± 0.08 | 0.51 ± 0.07 | 0.51 ± 0.06 | 0.84 ± 0.05 | 0.55 ± 0.13 | 0.59 / 0.72 |
| Random Forest, per pixel (baseline) | 0.84 ± 0.07 | 0.57 ± 0.03 | 0.51 ± 0.01 | 0.86 ± 0.06 | 0.60 ± 0.04 | – |

![[real_s2_kappa.png]]

Per-class F1 on 2017 held-out blocks (mean over splits):

| Model | forest | grassland | shrubland | artificial surface |
|---|---|---|---|---|
| U-Net | 0.87 | 0.63 | 0.27 | 0.25 |
| FCN | 0.83 | 0.53 | 0.21 | 0.29 |
| Custom (my_model.py) | 0.87 | 0.65 | 0.26 | 0.34 |
| timm ResNet-18 | 0.89 | 0.65 | 0.22 | 0.24 |
| U-Net 2016 → fine-tuned 2017 | 0.89 | 0.66 | 0.23 | 0.27 |
| Random Forest, per pixel (baseline) | 0.92 | 0.67 | 0.18 | 0.26 |

Parameters: U-Net 118,420 · FCN 30,916 · custom 29,604 · timm ResNet-18 11,301,636 · fine-tuned U-Net trains 45,108 of 118,420 (encoder frozen).

## What this shows
1. **The pipeline works on real data end to end**: real CRS, real polygons, 13-band L1C, nodata-free composites, five architectures including a custom file and a timm encoder, fine-tuning with a frozen encoder, prediction on another year and ONNX export (max difference to PyTorch 1.8e-6).
2. **On 1 km² of labels, deep learning does not beat a pixel Random Forest.** The Random Forest has the highest kappa (0.57 vs 0.40–0.51). That is expected with ~8,000 strongly correlated labelled pixels; CNNs need more area and more label diversity to use spatial context. This is a test of the mechanics, not a benchmark.
3. **Forest and grassland are learned well** (F1 0.83–0.92 and 0.53–0.67). Shrubland and artificial surface stay at F1 0.2–0.3 for every model, including the Random Forest; with 7–111 validation pixels per split the scores are also very uncertain.
4. **Results depend strongly on the split** (SD up to 0.19 in OA). Seed 3 held out almost no grassland or shrubland, which raises OA but lowers kappa. Single-split numbers on small areas should not be trusted; repeated spatial splits are the minimum.
5. **Transfer to 2016 worked once atmospheric bands were dropped**; most models scored about as well on 2016 as on 2017.
6. **Uncertainty is informative for most models**: timm ResNet-18 0.37 on correct vs 0.60 on wrong pixels. Not for the U-Net here (0.76 on both): it stayed uncertain everywhere, so its entropy map carries no signal on this data.
7. **Possible label error:** a bright patch in the east of the scene is predicted as artificial surface but labelled forest; it is clearly visible in the true-colour composite. Worth checking before treating it as a model error.

## Bugs and gaps the real data exposed (all fixed)
- **No way to drop label values** (empty no-data polygons, very rare classes) → added `labels.exclude`.
- **Atmospheric band B10 shifted +6.1 SD between years**, so U-Net transfer to 2016 collapsed to OA 0.44 while the Random Forest stayed at 0.91 → added [[check_band_shift]], which warns on prediction; configs now drop B01/B09/B10. See [[Band shift]].
- **Checkpoint selection by macro-F1 picked a barely trained epoch** (epoch 2, entropy 0.94 everywhere) because one class had 14 validation pixels → default is now validation loss (`train.select_metric`). See [[Checkpoint selection]].

## Reproduce
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/sentinel-hub/eo-learn.git
cd eo-learn && git sparse-checkout set example_data && cd ..
python examples/real_s2_slovenia.py eo-learn/example_data/TestEOPatch real_data
eoforge train configs/real_s2_timm.yaml
eoforge predict runs/real_s2_timm/best.pt real_data/s2_slovenia_2016_AprOct.tif --out pred_2016
python tools/real_test_report.py     # after the seed runs: tables and figures
```

## Next for a real benchmark
A KAZA scene with your land-cover points, or a larger public set (thousands of km²), would let the CNNs show whether spatial context helps. Area-adjusted accuracy ([[Roadmap]]) would also make the scores comparable with your published maps.
