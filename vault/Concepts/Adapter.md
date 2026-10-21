---
tags: [concept, zoo]
---
# Adapter

Code that knows how to run one kind of model. The runner ([[run_model]]) has already read the card's bands in the card's order, so an adapter receives `image (C, H, W)` in the image's own units, plus the valid-pixel mask and the raster profile, and returns a `Result`:

```python
from eoforge.zoo.adapters.base import Adapter, Result, register

@register("my-kind")
class MyAdapter(Adapter):
    requires = ("some_package",)          # checked before loading; the app shows card.install
    def load(self):
        ...                               # build the model once; self.device is cuda/cpu
    def predict(self, image, valid, profile, options=None):
        ...
        return Result(classes=classes_uint8, probs=probs_or_None)
```

Then add `from . import my_module` in `eoforge/zoo/adapters/__init__.py`, and set `adapter: my-kind` in a card. Helpers: [[tiled_probs]] (blended tiles), [[normalise]], `softmax`.

| Adapter | Runs | Notes |
|---|---|---|
| `onnx` | any ONNX segmentation model | fixed or dynamic input size; CUDA if `onnxruntime-gpu` is installed, else CPU |
| `terratorch` | TerraTorch configs + checkpoints (Prithvi) | uses the config's own scaling and normalisation; skips re-downloading the backbone |
| `omnicloudmask` | OmniCloudMask | red, green, NIR |
| `samgeo` | SAM / SAM 2 automatic masks | writes an 8-bit RGB view, returns object ids, also a GeoPackage |
| `terramind` | TerraMind any-to-any generation (LULC output) | tiles of 256 px; stochastic, seeded |
| `ftw` | Fields of The World checkpoints (segmentation-models-pytorch U-Net) | `params.windows: 2` for two dates; writes fields.gpkg |
| `delineate_anything` | Delineate Anything YOLO models (ultralytics) | overlapping tiles, polygons merged across tiles (FTW's method) |
| `eoforge` | your `best.pt` | normalisation from the checkpoint |
