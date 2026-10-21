---
tags: [module]
---
# eoforge.config

YAML configuration for a run. One file describes imagery, labels, model, training and outputs. Reference: [[Config reference]].

**Dataclasses:** `DataConfig`, `LabelConfig`, `ModelConfig`, `TrainConfig`, `PredictConfig`, and `Config` (top level, with `scenes` for multi-image runs and the `run_dir` property = `output_dir/name`). [[load_config]] checks each `scenes` entry (`image`, `labels`, optional `name`) and resolves its paths.

**Functions:** [[load_config]] · [[config_to_dict]]

**Private:** `_build(cls, raw, where)` builds nested dataclasses from dicts and raises on unknown keys, naming the section.
