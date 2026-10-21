---
tags: [function, pipeline]
module: eoforge.pipeline
---
# export_onnx

> Export a checkpoint to ONNX for offline and field use.

**Module:** [[eoforge.pipeline]] · **Source:** `eoforge/pipeline.py` · **Kind:** function

```python
export_onnx(checkpoint, out_path, base_dir=None, tile=256) -> Path
```

## How it works

Input `image` (B, C, tile, tile) with a dynamic batch axis; output `logits`. A JSON sidecar holds classes, band names and normalisation stats. Inputs must be normalised as `(x - mean) / std` and softmax applied to the logits. Verified to match PyTorch to ~2e-6.

## Used by

[[eoforge.cli]]
