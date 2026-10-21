---
tags: [function, zoo]
module: eoforge.zoo
---
# guess_descriptions

> Band names for images whose bands have none.

**Module:** [[eoforge.zoo]] · **Source:** `eoforge/zoo/bands.py` · **Kind:** function

```python
guess_descriptions(descriptions, required=None) -> (names, note or None)
```

| Bands in the image | Assumed |
|---|---|
| 13 | all Sentinel-2 L1C bands in order: B01-B08, B8A, B09, B10, B11, B12 (e.g. Sen1Floods11 chips) |
| 12 | Sentinel-2 L2A order without B10 |
| as many as the model needs | the model's own order (e.g. Fields of The World chips: B04, B03, B02, B08) |

Only used when no band has a description; the note is shown as a warning so a wrong guess is visible. Otherwise bands are chosen by number in the app or with `--bands`.

## Used by

[[run_model]] · [[eoforge.app]]
