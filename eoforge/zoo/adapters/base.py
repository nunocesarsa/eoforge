"""Adapter interface and shared helpers.

An adapter turns (card, image) into a Result. The runner has already selected the card's bands
in the card's order, so `image` is (C, H, W) float32 in the image's own units (e.g. Sentinel-2
reflectance x 10000). Adapters do their own normalisation, tiling and device handling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..card import ModelCard

ADAPTERS: dict[str, type["Adapter"]] = {}


def register(name: str):
    def deco(cls):
        ADAPTERS[name] = cls
        return cls
    return deco


@dataclass
class Result:
    classes: np.ndarray                         # (H, W) uint8/uint16/uint32 class codes or object ids
    probs: Optional[np.ndarray] = None          # (K, H, W) float32, optional
    extra: dict[str, np.ndarray] = field(default_factory=dict)   # more rasters to write, e.g. confidence
    kind: str = "classes"                       # classes | objects
    info: dict = field(default_factory=dict)


class Adapter:
    #: python packages needed, checked before loading so the app can show a clear message
    requires: tuple[str, ...] = ()

    def __init__(self, card: ModelCard, device: str = "auto", log: Callable[[str], None] | None = None):
        self.card = card
        self.device = resolve_device(device)
        self.log = log or (lambda m: None)

    def load(self):
        pass

    def prefetch(self):
        """Download weights ahead of time (default: load the model once, which downloads it)."""
        self.load()

    def predict(self, image: np.ndarray, valid: np.ndarray, profile: dict, options: dict | None = None) -> Result:
        raise NotImplementedError


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def missing_packages(adapter_cls) -> list[str]:
    import importlib.util
    return [p for p in adapter_cls.requires if importlib.util.find_spec(p.split("[")[0].replace("-", "_")) is None]


# ------------------------------------------------------------------ normalisation
def normalise(image: np.ndarray, spec: dict | None) -> np.ndarray:
    """Normalisations seen in published models. spec examples:
        {type: scale, factor: 0.0001}
        {type: standard, mean: [...], std: [...]}           per band
        {type: minmax, min: [...], max: [...]}
        {type: dynamic_world}                               Dynamic World's log-sigmoid
        {type: none}
    """
    spec = spec or {"type": "none"}
    t = spec.get("type", "none")
    x = image.astype("float32")
    if t == "none":
        return x
    if t == "scale":
        return x * float(spec.get("factor", 1.0)) + float(spec.get("offset", 0.0))
    if t == "standard":
        m = np.asarray(spec["mean"], "float32")[:, None, None]
        s = np.asarray(spec["std"], "float32")[:, None, None]
        return (x * float(spec.get("factor", 1.0)) - m) / s
    if t == "minmax":
        lo = np.asarray(spec["min"], "float32")[:, None, None]
        hi = np.asarray(spec["max"], "float32")[:, None, None]
        return np.clip((x - lo) / (hi - lo), 0, 1)
    if t == "dynamic_world":
        p = np.asarray(DW_NORM, "float32")
        x = np.log(x * 0.005 + 1)
        x = (x - p[:, 0, None, None]) / p[:, 1, None, None]
        x = np.exp(x * 5 - 1)
        return (x / (x + 1)).astype("float32")
    raise ValueError(f"Unknown normalisation type '{t}'")


# Dynamic World's per-band constants (log-transformed 30/70 percentiles), from github.com/google/dynamicworld
DW_NORM = [[1.7417268007636313, 2.023298706048351], [1.7261204997060209, 2.038905204308012],
           [1.6798346251414997, 2.179592821212937], [1.7734969472909623, 2.2890068333026603],
           [2.289154079164943, 2.6171674549378166], [2.382939712192371, 2.773418590375327],
           [2.3828939530384052, 2.7578332604178284], [2.1952484264967844, 2.789092484314204],
           [1.554812948247501, 2.4140534947492487]]


# ------------------------------------------------------------------ tiling
def window_weights(size: int, overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones((size, size), "float32")
    ramp = np.ones(size, "float32")
    r = np.linspace(0.05, 1.0, overlap, dtype="float32")
    ramp[:overlap], ramp[-overlap:] = r, r[::-1]
    return np.outer(ramp, ramp)


def starts(n: int, tile: int, step: int):
    if n <= tile:
        return [0]
    s = list(range(0, n - tile, step))
    s.append(n - tile)
    return s


def tiled_probs(fn, image: np.ndarray, valid: np.ndarray, k: int, tile=512, overlap=32, batch=4,
                pad_mode="reflect", multiple: int = 1, progress=None, fixed: bool = False) -> np.ndarray:
    """Run fn(batch (B, C, t, t) float32) -> probs (B, K, t, t) over the image with blended overlaps.

    `multiple` pads tiles up to a multiple of that size (e.g. 16 for ViTs) when the image is smaller.
    `fixed` keeps the tile size even for small images (models with a fixed input size); tiles are padded.
    """
    c, h, w = image.shape
    t = tile if fixed else min(tile, max(h, w))
    t = int(np.ceil(t / multiple) * multiple)
    step = max(1, t - 2 * overlap)
    wgt = window_weights(t, min(overlap, t // 4))
    acc = np.zeros((k, h, w), "float32")
    norm = np.zeros((h, w), "float32")
    wins = [(y, x) for y in starts(h, t, step) for x in starts(w, t, step)]
    done = 0
    for i in range(0, len(wins), batch):
        chips, metas = [], []
        for y, x in wins[i:i + batch]:
            chip = image[:, y:y + t, x:x + t]
            th, tw = chip.shape[1:]
            if not valid[y:y + th, x:x + tw].any():
                continue
            if (th, tw) != (t, t):
                mode = pad_mode if (pad_mode == "reflect" and th > 1 and tw > 1 and t - th < th and t - tw < tw) else "constant"
                chip = np.pad(chip, ((0, 0), (0, t - th), (0, t - tw)), mode=mode)
            chips.append(chip)
            metas.append((y, x, th, tw))
        if chips:
            out = fn(np.stack(chips).astype("float32"))
            for p, (y, x, th, tw) in zip(out, metas):
                ww = wgt[:th, :tw]
                acc[:, y:y + th, x:x + tw] += p[:, :th, :tw] * ww
                norm[y:y + th, x:x + tw] += ww
        done += len(wins[i:i + batch])
        if progress:
            progress(done / len(wins))
    norm[norm == 0] = 1
    return acc / norm


def softmax(x: np.ndarray, axis=1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
