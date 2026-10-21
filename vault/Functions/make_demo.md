---
tags: [function, demo]
module: eoforge.demo
---
# make_demo

> Create the synthetic demo data set.

**Module:** [[eoforge.demo]] · **Source:** `eoforge/demo.py` · **Kind:** function

```python
make_demo(out: Path)
```

## How it works

Writes `scene_2023.tif`, `labels_2023.gpkg` (~270 features), `truth_2023.npy`, and a second, slightly brighter scene of a different landscape (`scene_other.tif`, `truth_other.npy`) for transfer tests.

## Used by

[[eoforge.cli]]

## See also

[[Test - Synthetic demo]]
