---
tags: [concept, uncertainty]
---
# Entropy as uncertainty

For each pixel, [[entropy]] computes the Shannon entropy of the class probabilities and divides by `log(K)`:

- **0**: all probability on one class (confident)
- **1**: all classes equally likely

Written to `entropy.tif` by [[write_outputs]]. In both tests, wrong pixels had a clearly higher mean entropy than correct ones ([[Test - Synthetic demo]]: 0.16 vs 0.64), so the map is useful for directing field checks or new labels.

Caveat: entropy reflects the model's own confidence. Deep networks can be confidently wrong, especially on data unlike the training scene ([[Band shift]]); it is not a calibrated error probability.
