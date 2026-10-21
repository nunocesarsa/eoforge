---
tags: [function, registry]
module: eoforge.models.registry
---
# register_model

> Decorator that adds a builder under a short `model.arch` name.

**Module:** [[eoforge.models.registry]] · **Source:** `eoforge/models/registry.py` · **Kind:** decorator

```python
@register_model(name: str)
```

## How it works

```python
from eoforge.models.registry import register_model

@register_model("my_net")
def build(in_channels, num_classes, width=32):
    return MyNet(in_channels, num_classes, width)
```
The module must be imported before `build_model` runs; for one-off models, `file.py:fn` is simpler.
