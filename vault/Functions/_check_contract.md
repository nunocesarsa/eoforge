---
tags: [function, registry]
module: eoforge.models.registry
---
# _check_contract

> Run a `(1, c, 64, 64)` zero input through the model and check the output is `(1, k, 64, 64)`.

**Module:** [[eoforge.models.registry]] · **Source:** `eoforge/models/registry.py` · **Kind:** private function

```python
_check_contract(model, c, k, arch)
```

## Raises

- `RuntimeError`: readable message naming the architecture

## Used by

[[build_model]]

## See also

[[Model contract]]
