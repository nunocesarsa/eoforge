---
tags: [concept, zoo]
---
# Model card

A YAML file that tells EOForge everything needed to run a ready-made model. Read by [[load_cards]] from `eoforge/zoo/cards/`, `~/.eoforge/models/`, folders in `$EOFORGE_MODELS`, and `./models/`. A card with the same `id` in a later folder replaces an earlier one.

```yaml
id: kaza-landcover-v1                 # short, unique
name: KAZA land cover U-Net
adapter: onnx                         # onnx | terratorch | omnicloudmask | samgeo | eoforge
task: segmentation                    # segmentation | cloudmask | objects
description: One or two sentences shown in the app.
sensor: Sentinel-2 L2A
bands: [B02, B03, B04, B08, B11, B12] # in the order the model expects
resolution_m: 10                      # the app warns when an image differs by > 25%
classes:
  - {value: 0, name: water, colour: "#419bdf"}
  - {value: 1, name: woodland, colour: "#397d49"}
params:                               # adapter settings, see [[Adapter]]
  layout: NCHW
  normalise: {type: standard, mean: [...], std: [...]}
  output: logits
  tile: 512
  overlap: 32
weights: {path: /path/to/kaza.onnx}     # or {url: https://...} or {hf_repo: org/name, file: x.pt, files: [config.yaml]}
licence: CC-BY-4.0
source: https://...
citation: ...
notes: Anything a user should know.
install: pip install ...              # shown when a package is missing
```

Weights given as `url` or `hf_repo` are downloaded once to `~/.cache/eoforge/models/<id>/` ([[fetch]]).

Normalisation types ([[normalise]]): `none`, `scale` (factor, offset), `standard` (mean, std per band, optional factor applied first), `minmax`, `dynamic_world`.
