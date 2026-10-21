---
tags: [module]
---
# eoforge.pipeline

End-to-end steps.

**Public:** [[train]] · [[predict]] · [[load_model]] · [[export_onnx]] · [[check_band_shift]]

**Private:**
- `_device(name)`: `auto` picks CUDA, then Apple MPS, then CPU.
- `_log(msg)`: timestamped progress line.
- `_load_scenes(cfg, run)`: the image(s) and label masks of a run: `data.path` or every entry of `scenes`; checks that classes and band counts agree.
- `_pooled_stats(scenes)`: band mean and SD over the valid pixels of all images.
- [[_class_weights]]: loss weights per class.
- [[_load_weights]]: partial checkpoint loading for [[Fine-tuning]].
- `_evaluate(model, dl, k, ignore, classes, device, loss_fn)`: confusion matrix, metrics and mean loss over a validation loader.

[[predict]] picks bands by name when the new image has the training band names, so a different band order does not silently break a model.
