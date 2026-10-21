"""Model cards: one YAML file per ready-to-use model.

A card says where the weights are, which adapter runs them, which bands the model expects
(in order), and what the output classes are. Adding a model means adding a card; models that
need new code get an adapter in eoforge/zoo/adapters.

Cards are read from, in this order (later ones override earlier ones with the same id):
  1. eoforge/zoo/cards/           built-in cards
  2. ~/.eoforge/models/           your cards
  3. $EOFORGE_MODELS              extra folders, separated by ':' (or ';' on Windows)
  4. ./models/ in the working directory
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

BUILTIN = Path(__file__).parent / "cards"


@dataclass
class ClassInfo:
    value: int
    name: str
    colour: str = "#888888"


@dataclass
class ModelCard:
    id: str
    name: str
    adapter: str                          # onnx | terratorch | omnicloudmask | samgeo | eoforge
    task: str = "segmentation"            # segmentation | cloudmask | objects
    description: str = ""
    sensor: str = ""                      # e.g. "Sentinel-2 L1C"
    bands: list[str] = field(default_factory=list)   # required bands, in model order
    resolution_m: Optional[float] = None
    classes: list[ClassInfo] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)   # adapter settings
    weights: dict[str, Any] = field(default_factory=dict)  # url | hf_repo+file | path
    licence: str = ""
    source: str = ""
    citation: str = ""
    notes: str = ""
    install: str = ""                     # extra packages the adapter needs
    path: Optional[Path] = None           # card file location

    @property
    def folder(self) -> Path:
        return self.path.parent if self.path else BUILTIN

    def class_name(self, value: int) -> str:
        for c in self.classes:
            if c.value == value:
                return c.name
        return str(value)


def _parse(raw: dict, path: Path | None) -> ModelCard:
    raw = dict(raw)
    classes = []
    for i, c in enumerate(raw.pop("classes", []) or []):
        if isinstance(c, str):
            classes.append(ClassInfo(i, c))
        else:
            classes.append(ClassInfo(int(c.get("value", i)), str(c["name"]), str(c.get("colour", c.get("color", "#888888")))))
    known = set(ModelCard.__dataclass_fields__) - {"classes", "path"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}; allowed: {sorted(known)}")
    for req in ("id", "name", "adapter"):
        if req not in raw:
            raise ValueError(f"{path}: missing '{req}'")
    return ModelCard(classes=classes, path=path, **raw)


def card_dirs() -> list[Path]:
    dirs = [BUILTIN, Path.home() / ".eoforge" / "models"]
    extra = os.environ.get("EOFORGE_MODELS", "")
    for part in extra.replace(";", ":").split(":"):
        if part.strip():
            dirs.append(Path(part.strip()).expanduser())
    dirs.append(Path.cwd() / "models")
    return dirs


def load_cards(extra_dirs: list[Path] | None = None) -> dict[str, ModelCard]:
    cards: dict[str, ModelCard] = {}
    errors = []
    for d in card_dirs() + list(extra_dirs or []):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.y*ml")):
            try:
                c = _parse(yaml.safe_load(f.read_text()) or {}, f)
                cards[c.id] = c
            except Exception as e:      # a broken card should not hide the others
                errors.append(f"{f}: {e}")
    load_cards.errors = errors
    return cards


load_cards.errors = []


def get_card(model_id: str) -> ModelCard:
    cards = load_cards()
    if model_id not in cards:
        raise KeyError(f"No model card '{model_id}'. Available: {sorted(cards)}")
    return cards[model_id]


def write_card(card: dict, folder: Path | None = None) -> Path:
    """Save a card dict as YAML in the user's model folder (used by the app's 'Add a model' form)."""
    folder = folder or (Path.home() / ".eoforge" / "models")
    folder.mkdir(parents=True, exist_ok=True)
    _parse(card, None)                       # validate
    p = folder / f"{card['id']}.yaml"
    p.write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True))
    return p
