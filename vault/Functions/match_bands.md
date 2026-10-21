---
tags: [function, zoo]
module: eoforge.zoo
---
# match_bands

> Find the image bands a model needs, by name.

**Module:** [[eoforge.zoo]] · **Source:** `eoforge/zoo/bands.py` · **Kind:** function

```python
match_bands(required, descriptions, prefix=None) -> (indices, missing)
```

## How it works

Names are normalised first: `B2`, `b02` and `blue` all mean B02; `NIR` is B08; `nir_narrow` is B8A; `swir1` is B11. With `prefix`, only bands like `dry_B04` are used; [[run_model]] picks the first band set of a seasonal stack when no prefix is given, and says so. Bands without names are handled by [[guess_descriptions]] or chosen by number. Missing bands can have stand-ins in the card (`params.band_fallback`, e.g. `{B8A: B08}`).

## Used by

[[run_model]] · [[eoforge.app]]
