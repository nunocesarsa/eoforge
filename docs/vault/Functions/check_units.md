---
tags: [function, zoo]
module: eoforge.zoo
---
# check_units

> Bring an image to the units the model expects: Sentinel-2 style reflectance x 10000.

**Module:** [[eoforge.zoo]] · **Source:** `eoforge/zoo/run.py` · **Kind:** function

```python
check_units(img, valid, card, options) -> (img, notes)
```

## How it works

Works on a sample of up to 200,000 valid pixels.

- **0-1 reflectance**: if the 99th percentile is at most 2, the image is multiplied by 10000. Earth Engine and many tools export reflectance this way.
- **Offset**: `options['offset']` (e.g. 1000) is subtracted and values are clipped at 0. Sentinel-2 data processed with baseline 04.00 or later (from 25 January 2022) carry +1000 unless the provider removed it.
- **Offset warning**: when every band's 1st percentile is at least 1000, the data probably still carry the offset; the note suggests `offset 1000`. Very hazy or bright scenes can trigger it falsely, so it only warns.

Cards with `params.input: any` (SAM 2, OmniCloudMask, trained eoforge models) are not touched. Notes become warnings in `summary.json` and the app.

## Used by

[[run_model]]
