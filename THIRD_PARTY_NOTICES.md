# Third-party notices

EOForge's own code is MIT-licensed (see LICENSE). It includes or downloads the following.

## Included in this repository
| What | Where | Licence |
|---|---|---|
| Dynamic World model (Google LLC), converted to ONNX | `eoforge/zoo/cards/weights/dynamic_world_forward.onnx` | Apache-2.0, see `eoforge/zoo/cards/weights/DYNAMIC_WORLD_LICENSE.txt` and `DYNAMIC_WORLD_NOTICE.txt` |
| Polygon merging adapted from Fields of The World tools (kerner-lab) | `eoforge/zoo/adapters/fields.py` (`merge_polygons`) | MIT, Copyright (c) 2024 kerner-lab |
| Example U-Net trained on LandCoverNet Africa chip 33LYE_20 | `runs/example_unet_33LYE_20/` | model: MIT; training data LandCoverNet Africa (Radiant Earth Foundation), CC-BY-4.0 |

## Downloaded on first use (not redistributed here)
| Model | Licence |
|---|---|
| Prithvi-EO-2.0 fine-tunes (NASA/IBM) | Apache-2.0 |
| TerraMind 1.0 (IBM/ESA) | Apache-2.0 |
| Fields of The World v3.1 CC-BY checkpoints | CC-BY-4.0 (v2 single-window checkpoint: includes non-commercial training data) |
| Delineate Anything (weights and the ultralytics library) | **AGPL-3.0**: a service built on it must publish its source |
| OmniCloudMask | MIT |
| SAM 2 (Meta) / segment-geospatial | Apache-2.0 / MIT |

## Data used by the tools
LandCoverNet Africa (CC-BY-4.0), Sentinel-2 (Copernicus, free and open), ESA WorldCover 2021 (CC-BY-4.0),
Sen1Floods11 (Cloud to Street, see its repository), eo-learn example patch (MIT). Cite the sources when you publish results.
