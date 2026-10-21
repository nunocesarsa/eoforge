"""TerraMind (IBM / ESA) any-to-any generation, used here to generate a land-cover map from Sentinel-2 L2A.

TerraMind does not classify the image directly: it *generates* the land-cover modality it learned
from Esri's 10 m annual land cover (TerraMesh), conditioned on the Sentinel-2 input. IBM calls such
outputs "mental images": useful for a quick look, not a measurement.

Card params:
  variant: terramind_v1_base_generate   (tiny | small | base | large)
  timesteps: 10                          diffusion steps (fewer = faster, rougher)
  tile: 256, overlap: 32                 TerraMind was trained on 224 px; tiles must be multiples of 16
  seed: 0                                generation is stochastic; fixed seed for repeatable maps
The Sentinel-2 +1000 offset (processing baseline 04.00+) is handled by the runner (option offset: 1000).
"""
from __future__ import annotations

import numpy as np
import torch

from .base import Adapter, Result, register, tiled_probs


@register("terramind")
class TerraMindLULC(Adapter):
    requires = ("terratorch",)

    def load(self):
        from terratorch.registry import FULL_MODEL_REGISTRY
        p = self.card.params
        kw = {}
        self.log(f"Building {p.get('variant', 'terramind_v1_base_generate')} (first time downloads the weights) ...")
        self.model = FULL_MODEL_REGISTRY.build(
            p.get("variant", "terramind_v1_base_generate"), modalities=["S2L2A"], output_modalities=["LULC"],
            pretrained=bool(p.get("pretrained", True)), standardize=True, timesteps=int(p.get("timesteps", 10)), **kw)
        self.model = self.model.to(self.device).eval()
        self.log(f"TerraMind on {self.device}")

    @torch.no_grad()
    def predict(self, image, valid, profile, options=None):
        p = {**self.card.params, **(options or {})}
        import random
        seed = int(p.get("seed", 0))
        random.seed(seed)                # TerraMind draws its sampling seed from Python's random module
        np.random.seed(seed)
        torch.manual_seed(seed)
        x = image.astype("float32")
        x[:, ~valid] = 0

        def run(batch):
            out = self.model(torch.from_numpy(batch).to(self.device))["LULC"]
            return torch.softmax(out.float(), 1).cpu().numpy()

        tile = int(p.get("tile", 256))
        probs = tiled_probs(run, x, valid, 10, tile, int(p.get("overlap", 32)), int(p.get("batch", 1)),
                            multiple=16, progress=p.get("_progress"), fixed=True)
        return Result(classes=probs.argmax(0).astype("uint8"), probs=probs,
                      info={"note": "generated land cover (TerraMind 'mental image'), not a direct classification"})
