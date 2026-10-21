---
tags: [concept]
---
# Checkpoint format

`best.pt` and `last.pt` are dictionaries saved with `torch.save`:

| Key | Content |
|---|---|
| `state_dict` | Model weights |
| `epoch` | Epoch it was saved at |
| `arch`, `params` | How to rebuild the model ([[build_model]]); custom file paths are stored absolute |
| `in_channels`, `classes` | Input bands count and class names in id order |
| `bands`, `band_names` | Band selection and names |
| `mean`, `std` | Training [[Normalisation]] stats |
| `ignore_index` | Unlabelled value |
| `config` | The full run config |

Everything needed to apply the model is inside it: [[load_model]] and [[predict]] need nothing else. [[export_onnx]] writes the same metadata as a JSON sidecar.
