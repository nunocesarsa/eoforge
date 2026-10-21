---
tags: [module, data]
---
# eoforge.data.raster

Reading imagery, computing [[Normalisation]] stats and burning vector labels onto the image grid.

**`Scene`** dataclass: `image` (C, H, W float32), `valid` (H, W bool), `profile` (rasterio profile), `band_names`.

**Functions:** [[read_scene]] · [[band_stats]] · [[normalise]] · [[rasterise_labels]]
