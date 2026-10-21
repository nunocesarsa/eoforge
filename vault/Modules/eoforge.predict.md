---
tags: [module, inference]
---
# eoforge.predict

Whole-scene inference and GeoTIFF output.

**Functions:** [[predict_scene]] · [[entropy]] · [[write_outputs]]

**Private:**
- [[_window_weights]]: blending weights for [[Tiled inference]].
- [[_starts]]: tile start positions along one axis.
- `_colormap(k)`: colour table for `classes.tif` (10-colour palette, 255 transparent).
