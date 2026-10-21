---
tags: [guide, zoo, data]
---
# Tutorial

> One small area, public data for every model in the zoo, and the commands to run and check each one.

```bash
python tools/get_tutorial_data.py                         # default: farms and woodland north of Lusaka, Zambia
python tools/get_tutorial_data.py --lat -1.95 --lon 30.06 --year 2023 --out tutorial_rwanda
python tools/get_tutorial_data.py --text-only             # rewrite TUTORIAL.md from an earlier download
```
No account is needed. Then open `tutorial/TUTORIAL.md`. The data are kept out of git (`.gitignore`).

## What it downloads
| File | Source | For |
|---|---|---|
| `s2_start_*.tif` | Sentinel-2 L2A, clearest scene in `--start` (default March-April) | land cover, FTW window A |
| `s2_end_*.tif` | clearest in `--end` (June-August) | FTW window B, Delineate Anything, SAM 2 |
| `s2_burn_*.tif` | clearest in `--burn` (September-October, fire season) | Prithvi burn scars |
| `s2_cloudy_*.tif` | 15-90 % cloud scene in `--cloudy` (January-February), closest to 45 % | OmniCloudMask |
| `worldcover_2021.tif` | ESA WorldCover 10 m, tiles mosaicked onto the same grid | reference for land cover; labels for training |
| `sen1floods11/*_S2Hand.tif`, `*_LabelHand.tif` | Sen1Floods11 hand-labelled chips (`--flood-countries`, default Nigeria, Ghana, Somalia) | Prithvi floods, with water labels |
| `train_worldcover.yaml` | | a U-Net on the start and end scenes with WorldCover labels (classes under 0.5 % left out) |
| `TUTORIAL.md`, `tutorial.json` | | steps and what was downloaded |

Sentinel-2 comes from Earth Search (Element84, AWS open data): STAC search, only scenes that cover the whole area, ranked by the scene's cloud cover, then checked with the scene classification layer (SCL) inside the area. 12 bands (B01-B12 without B10) are read from the cloud-optimised GeoTIFFs for the area only, resampled to one 10 m UTM grid and named. The +1000 offset is removed when the metadata say it is there **and** the darkest pixels confirm it.

## Steps in TUTORIAL.md
1. Dynamic World on the green and the dry scene, compared with WorldCover.
2. TerraMind on the green scene.
3. Fields of The World with start and end (in the app).
4. Delineate Anything.
5. OmniCloudMask on the cloudy scene.
6. Prithvi burn scars with `--match-resolution`.
7. Prithvi floods on a Sen1Floods11 chip, scored against its labels (IoU).
8. SAM 2.
9. Train a U-Net on WorldCover labels and map the burn-season scene it never saw.

## Checking a result
`python tools/compare_maps.py RESULT REFERENCE` resamples the reference to the result's grid and prints, for each reference class, the share of its pixels in each result class; then each result class's best-matching reference class and the agreement after that matching. For binary maps (water) it gives IoU, precision and recall. Class schemes differ between models and references, so read the table rather than one number.

First run: [[Test - Tutorial Zambia]].
