"""Imagery sources.

Every source turns a DataConfig into a local, analysis-ready GeoTIFF that the
rest of the pipeline reads. Adding STAC or Earth Engine later means adding one
class here; training and prediction do not change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..config import DataConfig

SOURCES: dict[str, type["ImageSource"]] = {}


def register_source(name: str):
    def deco(cls):
        SOURCES[name] = cls
        return cls
    return deco


class ImageSource(ABC):
    def __init__(self, cfg: DataConfig, workdir: Path):
        self.cfg = cfg
        self.workdir = Path(workdir)

    @abstractmethod
    def prepare(self) -> Path:
        """Return the path of a local GeoTIFF holding the imagery."""


@register_source("local")
class LocalGeoTIFF(ImageSource):
    def prepare(self) -> Path:
        if not self.cfg.path:
            raise ValueError("data.path is required for source 'local'")
        p = Path(self.cfg.path)
        if not p.exists():
            raise FileNotFoundError(p)
        return p


@register_source("stac")
class STACSource(ImageSource):
    def prepare(self) -> Path:
        raise NotImplementedError(
            "STAC source is the next pipeline to add (search by AOI/date, cloud mask, "
            "composite, write GeoTIFF). Use source: local for now."
        )


@register_source("gee")
class EarthEngineSource(ImageSource):
    def prepare(self) -> Path:
        raise NotImplementedError(
            "Earth Engine source is planned (export an ee.Image to GeoTIFF). "
            "Use source: local for now."
        )


def get_source(cfg: DataConfig, workdir: Path) -> ImageSource:
    if cfg.source not in SOURCES:
        raise ValueError(f"Unknown data.source '{cfg.source}'. Available: {sorted(SOURCES)}")
    return SOURCES[cfg.source](cfg, workdir)
