---
tags: [function, sources]
module: eoforge.data.sources
---
# get_source

> Instantiate the source named in `data.source`.

**Module:** [[eoforge.data.sources]] · **Source:** `eoforge/data/sources.py` · **Kind:** function

```python
get_source(cfg: DataConfig, workdir: Path) -> ImageSource
```

## Raises

- `ValueError`: unknown source name

## Used by

[[train]]
