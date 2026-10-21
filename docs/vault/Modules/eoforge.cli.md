---
tags: [module, cli]
---
# eoforge.cli

The `eoforge` command (`main(argv=None)`).

| Command | Does | Calls |
|---|---|---|
| `eoforge train CONFIG` | Train, validate, map the image(s) | [[train]] |
| `eoforge predict CKPT IMAGE [--out DIR] [--config CFG] [--device D]` | Apply a trained model to a GeoTIFF | [[predict]] |
| `eoforge export CKPT [--out model.onnx]` | ONNX export | [[export_onnx]] |
| `eoforge demo [--out DIR]` | Make synthetic data | [[make_demo]] |
| `eoforge zoo list [--trained FOLDERS]` | List model cards | [[load_cards]] |
| `eoforge zoo info ID` | Show one card | |
| `eoforge zoo run ID\|best.pt IMAGE [--out] [--bands 2,3,4] [--prefix dry] [--offset 1000] [--match-resolution] [--probabilities] [--device]` | Run a model | [[run_model]] |
| `eoforge zoo download [IDS]` | Fetch weights now | `prefetch` |
| `eoforge zoo status` | Which models are ready | |
| `eoforge app [--folder F] [--port 8501]` | Start the web app (the next free port if 8501 is taken) | [[eoforge.app]] |

## Tools (`tools/`)
| Script | Does |
|---|---|
| `get_tutorial_data.py` | Public tutorial data for one area and `TUTORIAL.md` ([[Tutorial]]) |
| `compare_maps.py RESULT REFERENCE` | Cross-table of a result against a reference map; IoU for binary maps |
| `fetch_landcovernet.py` | Download LandCoverNet chips ([[LandCoverNet showcase]]) |
| `run_showcase.py` | One model per chip, or `--pooled` one model for all chips |
| `convert_dynamic_world.py` | How the Dynamic World ONNX file was made from Google's TensorFlow model |
| `real_test_report.py` | Figures of [[Test - Real Sentinel-2 Slovenia]] |
