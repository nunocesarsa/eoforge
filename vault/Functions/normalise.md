---
tags: [function, zoo]
module: eoforge.zoo
---
# normalise

> Apply a card's normalisation to raw band values.

**Module:** [[eoforge.zoo]] · **Source:** `eoforge/zoo.py` · **Kind:** function

```python
normalise(image, spec) -> np.ndarray
```

## How it works

Types: `none`, `scale` (`factor`, `offset`), `standard` (`mean`, `std`, optional `factor` first), `minmax` (`min`, `max`), `dynamic_world` (log-sigmoid with Dynamic World's per-band constants).

## See also

[[Model card]]
