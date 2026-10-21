---
tags: [module, zoo]
---
# eoforge.zoo

Ready-made models: cards, adapters, runner and previews. Guide: [[Model zoo and app]].

| File | Role |
|---|---|
| `card.py` | [[Model card]] dataclasses, [[load_cards]], `get_card`, `write_card` |
| `bands.py` | band names: [[match_bands]], [[guess_descriptions]], `canonical`, `prefixes` |
| `weights.py` | [[fetch]] weights from a path, URL or Hugging Face |
| `run.py` | [[run_model]], `read_input`, [[check_units]], [[grid_at_resolution]], `load_adapter`, `prefetch`, outputs and class areas |
| `preview.py` | PNG previews in EPSG:4326 for the map |
| `adapters/` | [[Adapter]] base class, [[tiled_probs]], [[normalise]], and the adapters below |
| `cards/` | built-in cards and the Dynamic World ONNX file with its licence and notice |

| Adapter | File | Models |
|---|---|---|
| `onnx` | `onnx_seg.py` | Dynamic World, any ONNX segmentation model (fixed or free input size, NCHW or NHWC) |
| `terratorch` | `terratorch_seg.py` | Prithvi-EO-2.0 floods and burn scars; checks the input scale against the model's normalisation |
| `terramind` | `terramind.py` | TerraMind generated land cover (seeded: Python, NumPy and torch) |
| `ftw` | `fields.py` | Fields of The World, field polygons |
| `delineate_anything` | `fields.py` | Delineate Anything (YOLO), merged field polygons |
| `omnicloudmask` | `cloudmask.py` | OmniCloudMask |
| `samgeo` | `sam.py` | SAM 2 via segment-geospatial |
| `eoforge` | `eoforge_ckpt.py` | your trained `best.pt`, with a band-shift warning ([[Band shift]]) |

`trained_cards(folders)` and `card_from_checkpoint(path)` turn eoforge `best.pt` files into cards.
