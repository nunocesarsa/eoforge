---
tags: [function, registry]
module: eoforge.models.registry
---
# build_model

> Turn `model.arch` into a network and check it follows the [[Model contract]].

**Module:** [[eoforge.models.registry]] · **Source:** `eoforge/models/registry.py` · **Kind:** function

```python
build_model(arch: str, in_channels: int, num_classes: int, params=None, base_dir=None) -> nn.Module
```

## Parameters

| Name | Type | Meaning |
|---|---|---|
| `arch` | `str` | `unet`, `fcn`, any registered name, `timm:<encoder>`, `file.py:fn` or `package.module:fn` |
| `params` | `dict` | keyword arguments for the builder |
| `base_dir` | `Path` | folder for relative `file.py` paths (the config's folder) |

## Returns

The built model.

## Raises

- `ValueError`: unknown name
- `RuntimeError`: forward pass fails or output shape is wrong

## Used by

[[train]] · [[load_model]]

## See also

[[Bring your own model]] · [[register_model]]
