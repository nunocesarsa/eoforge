---
tags: [function, zoo]
module: eoforge.zoo
---
# fetch

> Find or download a card's weights.

**Module:** [[eoforge.zoo]] · **Source:** `eoforge/zoo.py` · **Kind:** function

```python
fetch(card, progress=None) -> dict[str, Path]
```

## How it works

`path` (relative to the card file or absolute), `url` (downloaded once), or `hf_repo` + `file` (+ `files`) through huggingface_hub. Cache: `~/.cache/eoforge/models/<id>/`, or `$EOFORGE_CACHE`.

## See also

[[Model card]]
