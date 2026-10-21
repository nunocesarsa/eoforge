---
tags: [function, zoo]
module: eoforge.zoo
---
# load_cards

> Read all model cards from the card folders.

**Module:** [[eoforge.zoo]] · **Source:** `eoforge/zoo.py` · **Kind:** function

```python
load_cards(extra_dirs=None) -> dict[str, ModelCard]
```

## How it works

A card that cannot be read is skipped and its error kept in `load_cards.errors`; the app shows these errors on the *Models* tab.

## See also

[[Model card]]
