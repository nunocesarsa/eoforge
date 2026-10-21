"""EOForge model zoo: run ready-made models on your imagery without training.

    eoforge zoo list
    eoforge zoo run dynamic-world scene.tif
    eoforge app                                  # web interface

Python:
    from eoforge.zoo import list_models, run_model
    run_model("dynamic-world", "scene.tif")
"""
from __future__ import annotations

from pathlib import Path

from .card import ClassInfo, ModelCard, get_card, load_cards, write_card  # noqa: F401
from .run import BandError, load_adapter, prefetch, run_model  # noqa: F401

PALETTE = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def card_from_checkpoint(ck: Path) -> ModelCard | None:
    """A model card for one best.pt trained with eoforge (bands, classes and normalisation are inside it)."""
    import torch
    ck = Path(ck)
    try:
        meta = torch.load(ck, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if "classes" not in meta or "arch" not in meta:
        return None
    names = [str(n) for n in meta.get("band_names", [])]
    classes = [ClassInfo(i, c, PALETTE[i % len(PALETTE)]) for i, c in enumerate(meta["classes"])]
    run = ck.parent.name
    arch = meta["arch"]
    if ".py:" in arch:                                   # custom model file: show just its name
        arch = Path(arch.rsplit(":", 1)[0]).stem
    return ModelCard(
        id=f"trained-{run}", name=f"Trained: {run} ({arch})", adapter="eoforge", task="segmentation",
        description=f"Model you trained with eoforge ({meta['arch']}, best epoch {meta.get('epoch')}).",
        sensor="as in training", bands=names, classes=classes,
        params={"checkpoint": str(ck), "tile": 256, "overlap": 32, "input": "any"}, licence="yours", source=str(ck))


def trained_cards(search: list[Path] | None = None, max_depth: int = 5) -> dict[str, ModelCard]:
    """Cards for every best.pt (trained with eoforge) found under the search folders."""
    out = {}
    for root in search or [Path.cwd()]:
        root = Path(root)
        if not root.is_dir():
            continue
        for ck in sorted(root.rglob("best.pt")):
            if len(ck.relative_to(root).parts) > max_depth:
                continue
            c = card_from_checkpoint(ck)
            if c is not None:
                if c.id in out:                       # same run name in two places
                    c.id = f"{c.id}-{len(out)}"
                out[c.id] = c
    return out


def list_models(search_trained: list[Path] | None = None) -> dict[str, ModelCard]:
    cards = load_cards()
    if search_trained:
        cards.update(trained_cards(search_trained))
    return cards
