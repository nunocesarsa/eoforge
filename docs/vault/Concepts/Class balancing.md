---
tags: [concept, training]
---
# Class balancing

Land-cover labels are usually dominated by one or two classes. Two mechanisms counter that:

1. **Balanced chip sampling** in [[ChipDataset]]: chip centres are drawn so every class gets the same total probability, regardless of how many pixels it has.
2. **Weighted loss** via [[_class_weights]] with `class_weights: balanced`: inverse-frequency weights, rescaled to mean 1. Off by default since 0.6.0.

Both help rare classes but cannot create information that is not in the labels. Using both at once over-corrects: in [[Test - LandCoverNet showcase run 1]] rare classes such as water were over-predicted. The showcase therefore uses balanced sampling only. With only tens of pixels of a class (e.g. *artificial surface* in [[Test - Real Sentinel-2 Slovenia]]), scores for that class stay unreliable.
