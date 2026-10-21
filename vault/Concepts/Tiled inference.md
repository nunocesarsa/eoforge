---
tags: [concept, inference]
---
# Tiled inference

Scenes are too large to pass through a network at once. [[predict_scene]] slides a window of `tile_size` with a step of `tile_size - 2*overlap`, so neighbouring tiles overlap.

Each tile's softmax probabilities are multiplied by a weight that ramps down towards the tile edges ([[_window_weights]]) and accumulated; dividing by the summed weights gives a smooth blend without visible seams. Tiles with no valid pixels are skipped. The final window in each direction is aligned to the scene edge ([[_starts]]), so every pixel is covered.
