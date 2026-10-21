---
tags: [module, models]
---
# eoforge.models.registry

The "ingest any architecture" part. Resolves `model.arch` into a built, checked `nn.Module`.

**Functions:** [[build_model]] · [[register_model]] · [[encoder_parameters]] · [[_check_contract]]

**Registered names:** `unet` → [[UNet]], `fcn` → [[FCN]]

**Private:**
- `_timm_builder(encoder)` returns a builder that creates [[TimmSeg]]; imports timm lazily.
- `_load_callable(spec, base_dir)` loads `file.py:fn` (relative to `base_dir`) or `package.module:fn`.

Concept: [[Model contract]] · Guide: [[Bring your own model]]
