"""Model registry: the "ingest any architecture" part.

`model.arch` in the config can be:
  - a registered name:            "unet", "fcn"
  - a timm encoder:               "timm:resnet18"   (needs `pip install timm`)
  - your own builder in a file:   "my_models.py:build_model"
  - your own builder in a module: "package.module:build_model"

A builder is any callable `builder(in_channels, num_classes, **params) -> nn.Module`
whose model maps (B, C, H, W) to logits (B, K, H, W). That is the whole contract.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .builtin import FCN, UNet

MODELS: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    def deco(fn):
        MODELS[name] = fn
        return fn
    return deco


@register_model("unet")
def _unet(in_channels, num_classes, base=32, depth=3, dropout=0.1):
    return UNet(in_channels, num_classes, base=base, depth=depth, dropout=dropout)


@register_model("fcn")
def _fcn(in_channels, num_classes, width=64, layers=4):
    return FCN(in_channels, num_classes, width=width, layers=layers)


def _timm_builder(encoder: str):
    def build(in_channels, num_classes, pretrained=False, **_):
        try:
            from .timm_seg import TimmSeg
        except ImportError as e:  # pragma: no cover
            raise ImportError("arch 'timm:...' needs `pip install timm`") from e
        return TimmSeg(encoder, in_channels, num_classes, pretrained=pretrained)
    return build


def _load_callable(spec: str, base_dir: Path | None) -> Callable:
    target, _, attr = spec.rpartition(":")
    if target.endswith(".py"):
        p = Path(target)
        if not p.is_absolute() and base_dir is not None:
            p = base_dir / p
        mod_spec = importlib.util.spec_from_file_location(p.stem, p)
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"Cannot load model file {p}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    if not hasattr(module, attr):
        raise AttributeError(f"'{attr}' not found in {target}")
    return getattr(module, attr)


def build_model(arch: str, in_channels: int, num_classes: int, params: dict | None = None,
                base_dir: Path | None = None) -> nn.Module:
    params = dict(params or {})
    if arch in MODELS:
        builder = MODELS[arch]
    elif arch.startswith("timm:"):
        builder = _timm_builder(arch.split(":", 1)[1])
    elif ":" in arch:
        builder = _load_callable(arch, base_dir)
    else:
        raise ValueError(f"Unknown model.arch '{arch}'. Registered: {sorted(MODELS)}, "
                         "or use 'timm:<encoder>' or 'file.py:builder'")
    model = builder(in_channels=in_channels, num_classes=num_classes, **params)
    _check_contract(model, in_channels, num_classes, arch)
    return model


@torch.no_grad()
def _check_contract(model: nn.Module, c: int, k: int, arch: str):
    """Fail early, with a readable message, if a custom model breaks the contract."""
    was_training = model.training
    model.eval()
    x = torch.zeros(1, c, 64, 64)
    try:
        y = model(x)
    except Exception as e:
        raise RuntimeError(f"Model '{arch}' failed on a (1, {c}, 64, 64) input: {e}") from e
    finally:
        model.train(was_training)
    if not isinstance(y, torch.Tensor) or tuple(y.shape) != (1, k, 64, 64):
        shape = tuple(y.shape) if isinstance(y, torch.Tensor) else type(y)
        raise RuntimeError(f"Model '{arch}' must return logits of shape (B, {k}, H, W); got {shape}")


def encoder_parameters(model: nn.Module):
    """Parameters to freeze when model.freeze_encoder is true.

    Models can expose `encoder` (a module) to opt in; otherwise nothing is frozen.
    """
    enc = getattr(model, "encoder", None)
    return list(enc.parameters()) if isinstance(enc, nn.Module) else []
