"""YAML configuration for an eoforge run.

One file describes the whole experiment: where the imagery comes from,
which labels to learn from, which architecture to build, how to train it
and where to write outputs. Unknown keys raise an error so typos are caught.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    source: str = "local"                  # local | stac | gee (stac/gee: coming next)
    path: Optional[str] = None             # GeoTIFF for source=local
    bands: Optional[list[int]] = None      # 1-based band indices to use; None = all
    band_names: Optional[list[str]] = None
    nodata: Optional[float] = None         # overrides the raster's own nodata
    params: dict[str, Any] = field(default_factory=dict)  # source-specific (STAC/GEE later)


@dataclass
class LabelConfig:
    path: str = ""                          # vector file (gpkg, shp, geojson) or label raster (GeoTIFF)
    class_field: str = "class"              # attribute holding the class (int or string)
    classes: Optional[list[str]] = None     # order defines class ids; None = sorted unique values
    exclude: list[str] = field(default_factory=list)  # label values to drop (e.g. "", "no data", rare classes)
    point_buffer: float = 0.0               # metres; buffer points into small polygons
    class_map: Optional[dict[int, str]] = None  # raster labels: {code: class name}
    band: int = 1                           # raster labels: band holding the class codes
    ignore_index: int = 255


@dataclass
class ModelConfig:
    arch: str = "unet"                      # registry name, "timm:<encoder>", or "path/to/file.py:builder"
    params: dict[str, Any] = field(default_factory=dict)
    weights: Optional[str] = None           # optional checkpoint to start from (fine-tuning)
    freeze_encoder: bool = False


@dataclass
class TrainConfig:
    chip_size: int = 64
    chips_per_epoch: int = 512
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2               # fraction of spatial blocks held out
    block_size: int = 256                   # pixels; spatial blocks for the train/val split
    class_weights: str | list[float] | None = None    # None | balanced | explicit list (chips are already class-balanced)
    augment_radiometric: float = 0.1        # per-chip band gain/offset jitter (0 = off); helps transfer to other images
    val_scenes: Optional[list[str]] = None  # multi-scene runs: hold out these whole scenes (names) for validation
    augment: bool = True
    select_metric: str = "val_loss"         # best checkpoint by: val_loss | macro_f1 | overall_accuracy | kappa
    seed: int = 42
    device: str = "auto"
    num_threads: Optional[int] = None


@dataclass
class PredictConfig:
    tile_size: int = 256
    overlap: int = 32
    batch_size: int = 8
    write_probabilities: bool = False
    write_entropy: bool = True


@dataclass
class Config:
    name: str = "run"
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    # several images, each with its own labels: [{name: a, image: a.tif, labels: a_labels.tif}, ...]
    # (data.bands and the other labels.* settings apply to all of them)
    scenes: list[dict] = field(default_factory=list)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name


def _build(cls, raw: dict | None, where: str):
    raw = raw or {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown key(s) in '{where}': {sorted(unknown)}. Allowed: {sorted(known)}")
    kwargs = {}
    for name, value in raw.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else None
        if is_dataclass(default):
            kwargs[name] = _build(type(default), value, f"{where}.{name}" if where else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _build(Config, raw, "")
    # Resolve relative data paths against the config file's folder.
    base = path.parent
    if cfg.data.path and not Path(cfg.data.path).is_absolute():
        cfg.data.path = str((base / cfg.data.path).resolve())
    if cfg.labels.path and not Path(cfg.labels.path).is_absolute():
        cfg.labels.path = str((base / cfg.labels.path).resolve())
    for sc in cfg.scenes:
        unknown = set(sc) - {"name", "image", "labels"}
        if unknown or "image" not in sc or "labels" not in sc:
            raise ValueError(f"Each entry of 'scenes' needs image and labels (and optionally name); got {sorted(sc)}")
        for k in ("image", "labels"):
            if not Path(sc[k]).is_absolute():
                sc[k] = str((base / sc[k]).resolve())
        sc.setdefault("name", Path(sc["image"]).stem)
    if cfg.model.weights and not Path(cfg.model.weights).is_absolute():
        cfg.model.weights = str((base / cfg.model.weights).resolve())
    if not Path(cfg.output_dir).is_absolute():
        cfg.output_dir = str((base / cfg.output_dir).resolve())
    return cfg


def config_to_dict(cfg) -> dict:
    if is_dataclass(cfg):
        return {f.name: config_to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    return cfg
