"""Find or download a model's weights.

`card.weights` can be:
  path: model.onnx                      file next to the card (or absolute)
  url: https://.../model.onnx           downloaded once into the cache
  hf_repo: org/name, file: x.pt         Hugging Face Hub (needs huggingface_hub)
Extra files from the same Hugging Face repo can be listed in `files: [config.yaml, ...]`.

Cache: ~/.cache/eoforge/models/<card id>/  (override with EOFORGE_CACHE)
"""
from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

from .card import ModelCard


def cache_dir(card: ModelCard) -> Path:
    root = Path(os.environ.get("EOFORGE_CACHE", Path.home() / ".cache" / "eoforge" / "models"))
    d = root / card.id
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch(card: ModelCard, progress=None) -> dict[str, Path]:
    """Return {'main': path, <other file names>: path}. Downloads on first use."""
    w = card.weights or {}
    say = progress or (lambda msg: None)
    if "path" in w:
        p = Path(w["path"]).expanduser()
        if not p.is_absolute():
            p = card.folder / p
        if not p.exists():
            raise FileNotFoundError(f"Weights for '{card.id}' not found: {p}")
        return {"main": p}
    if "url" in w:
        dst = cache_dir(card) / w.get("file", Path(w["url"]).name)
        if not dst.exists():
            say(f"Downloading {w['url']} ...")
            tmp = dst.with_suffix(dst.suffix + ".part")
            with urllib.request.urlopen(w["url"]) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.rename(dst)
        return {"main": dst}
    if "hf_repo" in w:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError("This model downloads from Hugging Face: pip install huggingface_hub") from e
        out = {}
        names = [w["file"]] + list(w.get("files", []))
        for i, name in enumerate(names):
            say(f"Fetching {w['hf_repo']}/{name} (first time only) ...")
            p = Path(hf_hub_download(w["hf_repo"], name, revision=w.get("revision"),
                                     cache_dir=str(cache_dir(card))))
            out["main" if i == 0 else name] = p
        return out
    return {}       # adapters such as omnicloudmask/samgeo manage their own weights
