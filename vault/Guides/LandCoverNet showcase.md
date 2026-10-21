---
tags: [guide, test, real-data]
---
# LandCoverNet showcase

> Download about 10 chips of LandCoverNet Africa (not the whole dataset), train models on your GPU (one per chip, or one for all chips) and get a gallery.

**Dataset:** LandCoverNet Africa v1.0, Radiant Earth Foundation, CC-BY-4.0, hosted on Source Cooperative (source.coop/radiantearth/landcovernet). 1,980 chips of 256 × 256 px (2.56 km) at 10 m across 66 Sentinel-2 tiles, 2018. Labels were made by three annotators per chip; band 2 is their consensus score (0–100).

Classes: 1 water · 2 artificial bare ground · 3 natural bare ground · 4 snow/ice · 5 woody · 6 cultivated · 7 (semi) natural vegetation.

## Run
```bash
source .venv/bin/activate
python tools/fetch_landcovernet.py --n 10 --out showcase/landcovernet
python tools/run_showcase.py showcase/landcovernet                       # one model per chip
python tools/run_showcase.py showcase/landcovernet --pooled --holdout 3  # one model, 3 unseen chips
```
Per chip: `gallery.png`, `results.json`, one run per chip in `runs/`. Pooled: `gallery_pooled.png`, `results_pooled.json`, `runs/pooled/` with one prediction folder per chip. boto3 and matplotlib are installed with EOForge.

## What `fetch_landcovernet.py` does
1. Lists the bucket anonymously (no account) and finds the tile folders.
2. Puts southern Africa first (UTM zones 33–36, latitude bands J–L, around KAZA), then the rest in random order. `--tiles 35KNA 34KGD` picks specific tiles.
3. Per tile, reads the labels of `--candidates` chips (default 4) and keeps the one with the most varied classes; near single-class chips are skipped.
4. Reads the scene classification layer (SCL) of every 2018 date and, per season (`--seasons`, default `wet:11-4,dry:5-10`), picks `--dates` clear dates (default 3, at least 50% clear) spread through that season. Downloads 10 bands (B02–B08, B8A, B11, B12; B01 and B09 left out, see [[Band shift]]).
5. Writes one cloud-masked median composite per season, stacked into `<chip>_s2_2018.tif` (20 bands, `wet_B02` ... `dry_B12`), and the labels (`<chip>_labels.tif`); `--min-consensus 60` drops labels the annotators disagreed on. Seasons matter because the labels are annual: see [[Test - LandCoverNet showcase run 1]].
6. `manifest.json` records dates, clear fractions and class pixels per chip.

About 5–10 MB per chip. `--local <folder>` reads a local copy with the same layout instead.

## What `run_showcase.py` does
Writes a config per chip (raster labels with `class_map`, see [[Config reference]]), trains with [[train]] using class-balanced chip sampling without extra loss weights (`--class-weights balanced` adds them), and draws the gallery: dry-season image, labels, prediction with held-out OA/kappa, [[Entropy as uncertainty]].

Options: `--arch timm:resnet18`, `--params '{\"pretrained\": false}'`, `--epochs 60`, `--batch 64`, `--only 35KNA_07`.

`--pooled` writes one config (`configs/pooled.yaml`) with every chip under `scenes`, picks `--holdout` chips at random (seeded) as `train.val_scenes`, trains one model with statistics pooled over all chips, and labels each gallery row *training chip* or *unseen chip*.

## Read the results with care
Per-chip mode trains and validates each chip on itself (16 blocks of 64 px, 4 held out): it shows how well a model maps a landscape from labels in the same area, not how it transfers. Only the unseen chips of the pooled mode measure transfer, and with 7 training chips they are few: see [[Test - LandCoverNet pooled run]].

## Tested
- Download logic against local mocks of two folder layouts.
- Real run on your GTX 1070 on 2026-10-20: [[Test - LandCoverNet showcase run 1]]. It worked end to end but exposed the single-season composite bug, now fixed.
- Pooled run on 2026-10-21 with the wet/dry composites: [[Test - LandCoverNet pooled run]].

If a chip is skipped, the script prints why and shows file names it saw; `--inspect` prints the real layout of one tile.
