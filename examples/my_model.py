"""Example of bringing your own architecture.

Point a config at it with:   model: {arch: "../examples/my_model.py:build", params: {width: 48}}

The only contract: build(in_channels, num_classes, **params) returns an nn.Module
mapping (B, C, H, W) -> logits (B, num_classes, H, W). Exposing `self.encoder`
lets `freeze_encoder: true` freeze it during fine-tuning.
"""
import torch
from torch import nn


class SpectralSpatialNet(nn.Module):
    """1x1 'spectral' convs first (per-pixel band mixing), then dilated 3x3 'spatial' context."""

    def __init__(self, in_channels, num_classes, width=48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, 1), nn.BatchNorm2d(width), nn.GELU(),
            nn.Conv2d(width, width, 1), nn.BatchNorm2d(width), nn.GELU(),
        )
        self.context = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(width, width, 3, padding=d, dilation=d), nn.BatchNorm2d(width), nn.GELU())
            for d in (1, 2, 4)
        ])
        self.head = nn.Conv2d(width, num_classes, 1)

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x + self.context(x))


def build(in_channels, num_classes, width=48):
    return SpectralSpatialNet(in_channels, num_classes, width)
