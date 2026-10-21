---
tags: [guide, zoo]
---
# Model zoo and app

> Run ready-made models on your own imagery, see the result on a map, and add new models when you find them. No training or labels needed.

## Install everything (once)
```bash
cd eoforge
bash install.sh                      # or: bash install.sh /mnt/d/venvs/eoforge
```
This creates the environment (default `./.venv`), installs PyTorch 2.14 built for CUDA 12.6 (the last build that supports GTX 10xx cards), EOForge with every model library (TerraTorch, OmniCloudMask, segment-geospatial, SAM 2, ONNX Runtime, Streamlit), and downloads all model weights (~4 GB, once). `eoforge zoo status` shows what is ready; `eoforge zoo download` fetches weights again if a download failed.

Options: `SKIP_DOWNLOAD=1 bash install.sh`, `TORCH_CUDA=cu128 bash install.sh` (newer GPUs), `TMPDIR=/mnt/d/wsl_tmp bash install.sh` when the drive holding `/tmp` is small.

## Start the app
```bash
source .venv/bin/activate
eoforge app --folder /path/to/your/images
```
Open **http://localhost:8501** (under WSL in your Windows browser; WSL forwards it). If the port is taken, the app uses the next free one and says so. `Ctrl+C` in the terminal stops the app.

The app lists every GeoTIFF under `--folder`, and saves results to `<folder>/eoforge_results/<image>__<model>/`.

## The four steps (sidebar)
1. **Image**: pick a GeoTIFF from the folder, or upload one (saved to `<folder>/uploads`).
2. **Model**: built-in cards, your own cards, and every `best.pt` you trained under the folder. A warning shows the install command if a model needs a package you don't have.
3. **Bands**: matched by name automatically (B2 = B02 = blue, NIR = B08, ...; see [[match_bands]]); unnamed bands are read in Sentinel-2 or the model's order ([[guess_descriptions]]). Bands the image lacks show *not set*, and **Run** stays disabled until you choose them. Seasonal stacks (`wet_B04`, `dry_B04`) get a *Use bands from* choice.
4. **Run**: options for device, saving probabilities, tile size, *Subtract offset* (1000 for Sentinel-2 processed since 2022 when the provider kept the offset) and *Resample to the model's pixel size*. The expected input (sensor, level, pixel size) is shown under the model. Progress shows while the model runs on your GPU.

The result appears on a satellite basemap with an opacity slider, an optional uncertainty layer, a legend with the area of each class (% and ha), and download buttons.

## Models included
| Card id | Model | Download |
|---|---|---|
| `dynamic-world` | Google Dynamic World land cover, 9 classes, Sentinel-2 L1C, 10 m | included |
| `prithvi-floods` | Prithvi-EO-2.0 300M fine-tuned on Sen1Floods11: water/flood | 1.3 GB |
| `prithvi-burnscars` | Prithvi-EO-2.0 300M burn scars (HLS, 30 m) | 1.3 GB |
| `omnicloudmask` | OmniCloudMask clouds, thin clouds, shadows | small |
| `sam2` | Segment Anything 2 via segment-geospatial: objects, no classes | ~900 MB |
| `terramind-lulc` | TerraMind 1.0 base generates land cover in Esri's 10 classes from 12-band Sentinel-2 L2A (global, incl. Africa) | from Hugging Face on first use |
| `ftw-fields` | Fields of The World v3.1 (B5, CC-BY): field / boundary / not field from **two dates**, plus field polygons | 126 MB |
| `ftw-fields-fast` | the same with EfficientNet-B3 | 53 MB |
| `ftw-fields-1date` | FTW v2 single-date model (non-commercial training data) | 204 MB |
| `delineate-anything` | Delineate Anything v2: one polygon per field from true colour, 0.25-10 m (AGPL-3.0) | from Hugging Face on first use |
| `delineate-anything-small` | Delineate Anything v1 small, faster | from Hugging Face on first use |
| `trained-<run>` | your own eoforge models | – |

All are installed by `install.sh`. SAM 2's `sam2` package is built against your PyTorch without its optional CUDA extension (`SAM2_BUILD_CUDA=0 pip install --no-build-isolation sam2`), which avoids pip downloading a second, incompatible PyTorch.

## Two-date models (Fields of The World)
FTW compares the start and the end of the growing season. In the app, step 3 then shows **Start of season** and **End of season**: each picks an image and a band set. Seasonal stacks from `fetch_landcovernet.py` work directly (`wet` = start, `dry` = end in southern Africa); two separate images also work and are aligned automatically. From the command line or Python: `run_model(card, image, windows=[{"image": a}, {"image": b}])`.

FTW tends to mark pasture and paddocks as fields too (its authors suggest masking with a land-cover map). On a LandCoverNet chip in Botswana/South Africa it followed the fenced paddock grid.

## Bands a model needs but the image lacks
TerraMind needs all 12 L2A bands. Cards can name stand-ins (`params.band_fallback`, e.g. B01 → B02, B09 → B8A for TerraMind, B8A → B08 for OmniCloudMask and Prithvi); the app then shows which stand-ins are used and the summary records them as warnings.

## Input checks
Before every run [[run_model]] scales 0-1 reflectance to x10000, removes the offset when asked, and warns about pixel size, a likely 2022 offset, band shift (trained models) and an input scale that does not fit the model (TerraTorch). Warnings appear above the legend and in `summary.json`. Which data each model needs: [[Sample datasets per model]].

## Command line
```bash
eoforge zoo status                       # which models are ready
eoforge zoo download                     # fetch all weights now
eoforge zoo list                         # all model cards
eoforge zoo info dynamic-world           # one card
eoforge zoo run dynamic-world scene.tif  # run, results next to the image
eoforge zoo run sam2 drone.tif --bands 1,2,3
eoforge zoo run prithvi-burnscars s2.tif --match-resolution   # 10 m to the model's 30 m
eoforge zoo run terramind-lulc s2_2024.tif --offset 1000         # data that keep the 2022 offset
eoforge zoo run runs/example_unet_33LYE_20/best.pt scene.tif
```

## Adding a model
- **An ONNX segmentation model**: the app's *Add a model* tab writes a card for you. Fill in the file, bands in the model's order, normalisation and classes. Cards are saved in `~/.eoforge/models/`. See [[Model card]].
- **A PyTorch model**: export it to ONNX (`torch.onnx.export`), then add it as above. Your own eoforge models need nothing: they appear automatically.
- **A TerraTorch model** (other Prithvi fine-tunes, TerraMind, etc.): copy `eoforge/zoo/cards/prithvi_sen1floods11.yaml`, change `weights.hf_repo`, the checkpoint and config file names, bands and classes.
- **Something new** (a Hugging Face pipeline, an API): write an adapter, see [[Adapter]].

Tested end to end: an eoforge model exported to ONNX and added through a card gave the same map as the original checkpoint on 96.6% of pixels of a 100 px test image; the remainder is padding at the image edge.

## What was tested where
| Part | Tested in the cloud workspace | Needs your machine |
|---|---|---|
| Dynamic World | real model on real Sentinel-2 L1C (Slovenia) | – |
| Your trained models, ONNX cards | real models | – |
| Prithvi / TerraTorch | TerraTorch with an untrained Prithvi-EO-2.0 300M (no weights download) | the fine-tuned checkpoints |
| Fields of The World | real v3.1 B3 CC-BY weights on Slovenia (two years) and a LandCoverNet chip; app run with two band sets | – |
| TerraMind | untrained TerraMind tiny generator, incl. band stand-ins | the pretrained weights |
| Delineate Anything | an untrained YOLO11 segmentation model through the full path (tiles, polygons, merging, GeoPackage) | the trained weights |
| OmniCloudMask, SAM | the real libraries with a stand-in for the model call | the model weights |
| App | ran in a headless browser and with Streamlit AppTest: image, model, bands, run, legend | the basemap and map display (need internet) |

Run on your GTX 1070 since: Dynamic World on LandCoverNet and on the [[Tutorial]] scenes ([[Test - Tutorial Zambia]]), the pooled LandCoverNet U-Net ([[Test - LandCoverNet pooled run]]).
