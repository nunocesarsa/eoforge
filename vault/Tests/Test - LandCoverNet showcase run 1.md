---
tags: [test, real-data, lesson]
date: 2026-10-20
---
# Test: LandCoverNet showcase, run 1

> First real run of the [[LandCoverNet showcase]] on your GTX 1070. Everything ran end to end, but the maps were poor; the cause was found in the download step and fixed. Re-run pending.

![[showcase_run1_gallery.png]]

## Setup
- 10 chips of LandCoverNet Africa v1.0, all in southern Africa (Namibia, Botswana, Zambia, Zimbabwe, Angola), 256 × 256 px at 10 m, 2018, labels as GeoTIFF read through `class_map` ([[rasterise_labels]]).
- One U-Net per chip (base 32, depth 3), 40 epochs, 512 chips of 64 px per epoch, batch 32, on CUDA.
- Validation: 16 blocks of 64 px per chip, 4 held out ([[Spatial block validation]]).

## Results
| Chip | Dates used | OA | Kappa | Macro-F1 |
|---|---|---|---|---|
| 33KXV_17 | 01/02 – 01/15 | 0.42 | 0.17 | 0.32 |
| 35KNT_09 | 01/05 – 03/21 | 0.42 | 0.16 | 0.33 |
| 33LUC_01 | 01/03 – 03/14 | 0.40 | 0.10 | 0.18 |
| 35LNF_00 | 04/23 – 10/10 | 0.90 | 0.82 | 0.71 |
| 35JKL_12 | 01/03 – 03/04 | 0.26 | 0.04 | 0.22 |
| 33KVU_04 | 01/15 – 03/11 | 0.40 | 0.17 | 0.30 |
| 34LHJ_15 | 01/06 – 12/27 | 0.71 | 0.56 | 0.66 |
| 33LYE_20 | 01/15 – 05/10 | 0.86 | 0.75 | 0.48 |
| 35JNN_26 | 01/02 – 01/20 | 0.62 | 0.31 | 0.41 |
| 35LMC_06 | 01/08 – 10/05 | 0.50 | 0.32 | 0.42 |

Overall: mean OA 0.55, mean kappa 0.34 (range 0.04–0.82). Training-pixel accuracy was also low (0.33–0.95): the models could not fit even their own training labels.

## What went wrong
1. **Composites covered only part of the year (bug, fixed).** The script took "the 6 clearest dates"; in this dry region many dates are 100% clear, so ties were broken by date order and most chips got only January–March (33KXV_17: six dates within two weeks of January). LandCoverNet labels are annual, assigned from the full 2018 time series; *cultivated*, *semi-natural vegetation* and *woody* are mainly separated by phenology, which a single wet-season composite does not show. The three best chips (35LNF_00, 34LHJ_15, 33LYE_20, κ 0.56–0.82) are the ones whose dates happened to span more of the year: consistent with the diagnosis, though 10 chips are not proof.
2. **Double class balancing (likely, changed).** Chips were sampled class-balanced *and* the loss was inverse-frequency weighted, which over-corrects towards rare classes, e.g. water predicted along dry drainage lines in 33LUC_01 where labels have 204 water pixels. See [[Class balancing]].
3. **Blocky labels (data, not fixable here).** In 33KXV_17, 33LUC_01 and 33KVU_04 the labels form 32 × 32 squares: the annotation blocks the dataset documentation warns about. This is label noise. `--min-consensus` can drop pixels the three annotators disagreed on.

Pipeline mechanics were fine: images and labels share the same grid in every chip, reflectance ranges are as expected, and the visible features (water, river, fields, woodland) line up with the labels.

## Fixes (v0.3.1)
- `fetch_landcovernet.py` builds **one composite per season** (default wet Nov–Apr and dry May–Oct, 3 dates each, spread through the season) and stacks them: 20 bands named `wet_B02` ... `dry_B12`. `--seasons year:1-12` gives a single composite.
- `run_showcase.py` uses class-balanced sampling only (`--class-weights balanced` restores the old behaviour); the gallery shows the dry-season image.

## Re-run
```bash
mv showcase/landcovernet showcase/landcovernet_run1     # keep run 1
python tools/fetch_landcovernet.py --n 10 --out showcase/landcovernet
python tools/run_showcase.py showcase/landcovernet
```
