"""TerraTorch segmentation models (Prithvi-EO fine-tunes and anything else TerraTorch can build).

The model's own config decides scaling and normalisation: we build the config's datamodule and
apply exactly its `constant_scale` and normalisation (`aug`), as TerraTorch's inference does.

Card params:
  tile: 512, overlap: 64, batch: 1
  scale: 0.0001           input scale; the card wins over the datamodule's constant_scale, whose default
                          (1) only means "the training files were already in model units"
  backbone_pretrained: false   avoid re-downloading backbone weights (the checkpoint has them)
  use_location: false     pass the image centre as location_coords (Prithvi 'TL' models; optional)
weights: hf_repo + file (checkpoint) + files: [<config yaml>]   or   path (checkpoint) + params.config
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..weights import fetch
from .base import Adapter, Result, register, tiled_probs


def build_inference_model(config: Path, checkpoint: Path | None, no_backbone_download=True):
    """Build the TerraTorch task + datamodule from a config, then load the checkpoint.

    The fine-tuned checkpoint contains the backbone weights, so the config's
    `backbone_pretrained: true` (which would download ~1 GB again) is switched off in a temporary
    copy of the config. Training-only settings (mixed precision, loggers) are removed too.
    """
    import tempfile
    import yaml
    from terratorch.cli_tools import LightningInferenceModel
    cfg = yaml.safe_load(Path(config).read_text())
    margs = cfg.get("model", {}).get("init_args", {}).get("model_args", {})
    if no_backbone_download and "backbone_pretrained" in margs:
        margs["backbone_pretrained"] = False
    tr = cfg.setdefault("trainer", {})
    tr["precision"] = 32          # GTX 10xx have no fast fp16; inference in fp32
    tr["logger"] = False
    tr.pop("callbacks", None)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        return LightningInferenceModel.from_config(str(tmp), str(checkpoint) if checkpoint else None)
    finally:
        Path(tmp).unlink(missing_ok=True)


@register("terratorch")
class TerraTorchSegmentation(Adapter):
    requires = ("terratorch",)

    def load(self):
        files = fetch(self.card, self.log)
        p = self.card.params
        ckpt = files.get("main")
        cfg = None
        for name, path in files.items():
            if name != "main" and str(name).endswith((".yaml", ".yml")):
                cfg = path
        cfg = Path(p["config"]) if p.get("config") else cfg
        if cfg is None:
            raise ValueError(f"Card '{self.card.id}' needs a config yaml (weights.files or params.config)")
        if not Path(cfg).is_absolute() and self.card.path:
            cfg = self.card.folder / cfg if (self.card.folder / cfg).exists() else cfg
        self.log("Building TerraTorch model from config (first time can take a minute) ...")
        lim = build_inference_model(Path(cfg), Path(ckpt) if ckpt else None, p.get("backbone_pretrained", False) is False)
        self.task = lim.model.to(self.device).eval()
        self.dm = lim.datamodule
        if "scale" in p:
            self.scale = float(p["scale"])
        else:
            self.scale = float(getattr(self.dm, "constant_scale", None) or 1.0)
        self.aug = getattr(self.dm, "aug", None)
        self.log(f"TerraTorch model on {self.device}; input scale {self.scale:g}; "
                 f"normalisation {'from datamodule' if self.aug is not None else 'none'}")

    @torch.no_grad()
    def predict(self, image, valid, profile, options=None):
        p = {**self.card.params, **(options or {})}
        x = image.astype("float32") * self.scale
        x[:, ~valid] = 0
        loc = _location(profile) if p.get("use_location", False) else None

        def run(batch):
            t = torch.from_numpy(batch).to(self.device)
            if self.aug is not None:
                t = self.aug({"image": t})["image"]
            try:
                out = self.task(t, location_coords=loc.to(self.device).expand(t.shape[0], -1)) if loc is not None else self.task(t)
            except TypeError:
                out = self.task(t)
            logits = getattr(out, "output", out)
            return torch.softmax(logits.float(), 1).cpu().numpy()

        warnings = self._check_scale(x, valid)
        k = len(self.card.classes) or 2
        probs = tiled_probs(run, x, valid, k, int(p.get("tile", 512)), int(p.get("overlap", 64)),
                            int(p.get("batch", 1)), multiple=16, progress=p.get("_progress"))
        values = np.asarray([c.value for c in self.card.classes] or list(range(k)))
        return Result(classes=values[probs.argmax(0)].astype("uint8"), probs=probs, info={"warnings": warnings})

    def _check_scale(self, x, valid):
        """After scaling and the model's normalisation, band means should sit within a few SDs of 0."""
        if self.aug is None or not valid.any():
            return []
        try:
            ys, xs = np.nonzero(valid)
            sel = np.random.default_rng(0).choice(ys.size, min(ys.size, 4096), replace=False)
            sample = x[:, ys[sel], xs[sel]][None, :, :, None]           # (1, C, N, 1)
            z = self.aug({"image": torch.from_numpy(np.ascontiguousarray(sample))})["image"]
            m = z.float().mean(dim=(0, 2, 3)).abs().max().item()
        except Exception:
            return []
        if m > 8:
            return [f"After scaling by {self.scale:g} the input is {m:.0f} SDs away from the model's training data: "
                    "the image units probably differ from what the model expects (check params.scale)."]
        return []


def _location(profile):
    """(lat, lon) of the image centre, used by Prithvi 'TL' (temporal-location) models when given."""
    try:
        from rasterio.warp import transform as warp
        h, w, tf = profile["height"], profile["width"], profile["transform"]
        x, y = tf * (w / 2, h / 2)
        lon, lat = warp(profile["crs"], "EPSG:4326", [x], [y])
        return torch.tensor([[lat[0], lon[0]]], dtype=torch.float32)
    except Exception:
        return None
