---
tags: [guide, zoo, data]
---
# Sample datasets per model

A model only shows what it can do on data like its training data: same sensor, processing level, pixel size, and season. Each entry gives a test set **with reference labels** (so a result can be checked, not just looked at) and how to get your own matching image. The checks in [[run_model]] catch units, the 2022 offset and pixel size, but not the processing level or the season.

| Model | Needs | Test set with labels | Your own image |
|---|---|---|---|
| `dynamic-world` | S2 **L1C**, 9 bands, 10 m | Dynamic World test tiles and expert validation tiles (Zenodo) | Earth Engine `COPERNICUS/S2_HARMONIZED`; compare with `GOOGLE/DYNAMICWORLD/V1` of the same date |
| `terramind-lulc` | S2 **L2A**, 12 bands | TerraMesh `val` split (S2L2A + Esri LULC pairs) | Any L2A; reference: Esri/Impact Observatory 10 m annual LULC |
| `ftw-fields*` | S2 L2A, B04 B03 B02 B08, **two dates** | Fields of The World: Kenya, Rwanda, South Africa (+ 21 more countries) | Two L2A images: start and end of the season |
| `delineate-anything*` | RGB, 0.25-10 m | FTW chips (window B), AI4Boundaries (S2 + 1 m orthophotos) | Drone or aerial RGB (`scale: 255`) or S2 L2A |
| `prithvi-floods` | S2 **L1C**, 6 bands, 10 m | Sen1Floods11 hand-labelled S2 chips | L1C of a flood (e.g. Beira, Mozambique, March 2019) |
| `prithvi-burnscars` | **HLS** 30 m, 6 bands | HLS Burn Scars (US) | HLS S30/L30 from NASA Earthdata, southern-African fire season; reference MODIS MCD64A1 |
| `omnicloudmask` | red, green, NIR, 10-50 m | CloudSEN12 (same 4 classes) | Any cloudy S2 L1C/L2A or Landsat scene |
| `sam2` | any RGB, best < 2 m | none needed (objects, no classes) | OpenAerialMap drone/aerial imagery |
| trained example U-Net | LandCoverNet 10/20-band stacks | LandCoverNet chips | Train on several chips instead ([[LandCoverNet showcase]], `--pooled`) |

## Notes per model

**Dynamic World.** Trained on Sentinel-2 L1C (top of atmosphere). L2A images such as the LandCoverNet composites run but come out darker in blue and green, which pushes pixels towards *bare* and *shrub*. Open savanna woodland is usually *shrub and scrub* in Dynamic World's scheme, not *trees*: on LandCoverNet tile 35KNT_09, 74% of the pixels labelled *woody* came out as shrub. That is a class-definition difference, not a bug. The best check is to export the L1C bands and the Dynamic World label of the same date from Earth Engine and compare.
- Test tiles: https://zenodo.org/record/4766508 ; expert consensus tiles: https://zenodo.org/records/4766451 ; training data: https://doi.pangaea.de/10.1594/PANGAEA.933475

**TerraMind.** Generates Esri-style land cover from all 12 L2A bands. LandCoverNet lacks B01 and B09 (stand-ins are used). TerraMesh has matching S2L2A/LULC pairs: https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh

**Fields of The World.** Each chip has `s2_images/window_a/*.tif` and `window_b/*.tif` (4 bands, unnamed: EOForge reads them in the model's order R, G, B, NIR; the preview should look like natural colour) and `label_masks/semantic_3class/*.tif` to compare with. In the app choose window_a as *Start of season* and window_b as *End of season*. Per-country zips: https://source.coop/kerner-lab/fields-of-the-world (Rwanda alone: https://source.coop/kerner-lab/fields-of-the-world-rwanda). Kenya and Rwanda labels are presence-only (fields marked, non-fields not), so scores there are optimistic for background.

**Delineate Anything.** Same FTW chips (one window), or AI4Boundaries for 1 m orthophotos: https://essd.copernicus.org/articles/15/317/2023/

**Prithvi floods.** Sen1Floods11 (`gsutil -m rsync -r gs://sen1floods11 <folder>`, ~14 GB, Google Cloud SDK needed; S2Hand chips are 512 x 512, 13 unnamed bands, which EOForge reads as the L1C band order; labels: -1 no data, 0 not water, 1 water). Includes an event in Nigeria. https://github.com/cloudtostreet/Sen1Floods11

**Prithvi burn scars.** Trained on HLS at 30 m over the US: https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars (2.65 GB, 512 x 512, 6 bands). For a 10 m Sentinel-2 image tick *Resample to the model's pixel size*. Before 0.6.0 this model got inputs 10000x too large ([[Review - generalisation 0.6.0]]).

**OmniCloudMask.** CloudSEN12 uses the same classes (clear, thick, thin, shadow): https://huggingface.co/datasets/tacofoundation/cloudsen12

**SAM 2.** OpenAerialMap has free drone and aerial imagery of many African towns and farms: https://openaerialmap.org . Use bands 1, 2, 3 for an RGB GeoTIFF.

## Sentinel-2 checklist
- **L1C or L2A?** Dynamic World and Prithvi floods: L1C. TerraMind, FTW, Prithvi burn scars (HLS is surface reflectance): L2A.
- **Offset.** Data processed with baseline 04.00+ (from 25 Jan 2022) carry +1000 unless the provider removed it. Earth Engine `*_HARMONIZED` collections removed it; Planetary Computer did not; for other providers check their notes on `BOA_ADD_OFFSET`. Use *Subtract offset 1000* for those.
- **Units.** 0-1 floats are scaled automatically.
- **Season.** Fields models need start and end of the growing season; land cover is best checked against a reference from the same year.
