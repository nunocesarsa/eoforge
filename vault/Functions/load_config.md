---
tags: [function, config]
module: eoforge.config
---
# load_config

> Read a YAML file into a validated `Config`.

**Module:** [[eoforge.config]] · **Source:** `eoforge/config.py` · **Kind:** function

```python
load_config(path: str | Path) -> Config
```

## Parameters

| Name | Type | Meaning |
|---|---|---|
| `path` | `str | Path` | YAML file |

## Returns

`Config` with nested dataclasses. Relative `data.path`, `labels.path`, `model.weights` and `output_dir` are resolved against the config file's folder.

## Raises

- `ValueError`: unknown key in any section, with the section name and allowed keys

## Used by

[[eoforge.cli]] · [[train]]

## See also

[[Config reference]] · [[config_to_dict]]
