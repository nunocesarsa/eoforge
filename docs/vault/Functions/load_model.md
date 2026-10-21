---
tags: [function, pipeline]
module: eoforge.pipeline
---
# load_model

> Rebuild a model from a checkpoint and load its weights (eval mode).

**Module:** [[eoforge.pipeline]] · **Source:** `eoforge/pipeline.py` · **Kind:** function

```python
load_model(checkpoint, device='cpu', base_dir=None) -> (model, ckpt)
```

## Used by

[[predict]] · [[export_onnx]]

## See also

[[Checkpoint format]]
