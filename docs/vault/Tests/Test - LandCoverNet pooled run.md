---
tags: [test, real-data]
---
# Test: LandCoverNet pooled run

One U-Net (base 32, depth 3, dropout 0.2) trained on 7 LandCoverNet Africa chips at once and validated on 3 chips it never saw, on the GTX 1070, 2026-10-21. Command: `python tools/run_showcase.py showcase/landcovernet --pooled --holdout 3` (defaults: 40 epochs, 1,024 chips of 64 px per epoch). Input: wet and dry season composites, 20 bands. Guide: [[LandCoverNet showcase]].

![[landcovernet_pooled_gallery.png]]

## Results
| Chip | Role | OA | Kappa |
|---|---|---|---|
| 33LYE_20 | training | 0.93 | 0.85 |
| 35LNF_00 | training | 0.89 | 0.76 |
| 35JNN_26 | training | 0.67 | 0.35 |
| 35KNT_09 | training | 0.65 | 0.39 |
| 35JKL_12 | training | 0.62 | 0.08 |
| 33LUC_01 | training | 0.57 | 0.06 |
| 33KVU_04 | training | 0.55 | 0.18 |
| 34LHJ_15 | **unseen** | 0.64 | 0.35 |
| 33KXV_17 | **unseen** | 0.46 | 0.10 |
| 35LMC_06 | **unseen** | 0.37 | 0.01 |
| all unseen chips | | **0.49** | **0.23** (macro-F1 0.31) |

The best epoch was **3 of 40**: validation loss was lowest after epoch 1-3 (1.59) and rose to 2.5 while training loss kept falling (1.52 → 0.68).

## What it shows
- **Transfer is the hard part.** On its own chip a U-Net reaches kappa 0.85-0.93 ([[Test - LandCoverNet showcase run 1]]); on new chips the same kind of model gets 0.23. The per-chip scores were not wrong, they answered a different question.
- **It overfits within a few epochs.** With 7 chips the model learns the chips, not the classes. More chips (`fetch_landcovernet.py --n 40`), fewer epochs, a pretrained encoder (`--arch timm:resnet18` with `pretrained: true`) and stronger radiometric jitter are the next things to try.
- **It falls back to the dominant class of each landscape**: cultivated on 35JKL_12, natural bare ground on 33LUC_01, semi-natural vegetation on 35LMC_06. The uncertainty map is bright (high entropy) exactly there, so entropy flags where not to trust the map.
- **Labels vary between chips.** Several chips (33KXV_17, 33LUC_01, 33KVU_04) have blocky patches of labels; class definitions such as *woody* vs *semi-natural vegetation* are hard to tell apart in dry savanna.

Next run: more chips, `--epochs 15`, compare with a timm encoder.
