"""Run a model trained with `eoforge train` (a best.pt checkpoint).

The checkpoint already knows its bands, classes and normalisation, so a card for it can be
generated automatically (see zoo.trained_cards); `params.checkpoint` points at the file.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import Adapter, Result, register, tiled_probs


@register("eoforge")
class EoforgeCheckpoint(Adapter):
    def load(self):
        from ...pipeline import load_model
        ckpt = self.card.params["checkpoint"]
        self.model, self.meta = load_model(ckpt, self.device)
        self.mean = np.asarray(self.meta["mean"], "float32")[:, None, None]
        self.std = np.asarray(self.meta["std"], "float32")[:, None, None]
        self.log(f"eoforge checkpoint {ckpt} on {self.device}")

    @torch.no_grad()
    def predict(self, image, valid, profile, options=None):
        p = {**self.card.params, **(options or {})}
        x = (image - self.mean) / self.std
        x[:, ~valid] = 0

        def run(batch):
            out = self.model(torch.from_numpy(batch).to(self.device))
            return torch.softmax(out, 1).cpu().numpy()

        warnings = []
        if valid.any():
            shift = (image[:, valid].mean(1) - self.mean[:, 0, 0]) / self.std[:, 0, 0]
            names = self.meta.get("band_names") or [str(i + 1) for i in range(len(shift))]
            far = [f"{n} {s:+.1f}" for n, s in zip(names, shift) if abs(s) > 2]
            if far:
                warnings.append("Band means differ from the training image by more than 2 SD (" + ", ".join(far) +
                                "); the model may not transfer well to this image.")
        k = len(self.meta["classes"])
        probs = tiled_probs(run, x.astype("float32"), valid, k, int(p.get("tile", 256)), int(p.get("overlap", 32)),
                            int(p.get("batch", 8)), progress=p.get("_progress"))
        return Result(classes=probs.argmax(0).astype("uint8"), probs=probs, info={"warnings": warnings})
