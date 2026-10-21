---
tags: [function, pipeline]
module: eoforge.pipeline
---
# _load_weights

> Load the tensors of a checkpoint that match by name and shape; skip the rest.

**Module:** [[eoforge.pipeline]] · **Source:** `eoforge/pipeline.py` · **Kind:** private function

```python
_load_weights(model, path) -> list[str]
```

## Returns

Names of skipped tensors (also logged with counts).

## Used by

[[train]]

## See also

[[Fine-tuning]]
