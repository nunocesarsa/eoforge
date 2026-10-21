---
tags: [function, registry]
module: eoforge.models.registry
---
# encoder_parameters

> Parameters of `model.encoder`, frozen when `freeze_encoder: true`.

**Module:** [[eoforge.models.registry]] · **Source:** `eoforge/models/registry.py` · **Kind:** function

```python
encoder_parameters(model) -> list[Parameter]
```

## Returns

Parameter list, or empty if the model has no `encoder` module (then nothing is frozen).

## Used by

[[train]]

## See also

[[Fine-tuning]]
