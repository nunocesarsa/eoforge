"""Any timm CNN encoder + a light FPN-style decoder, with arbitrary input bands.

timm adapts the first conv to `in_chans` (it repeats/averages RGB weights when
pretrained), so Sentinel-2 or Landsat band stacks work out of the box.
"""
from __future__ import annotations

import timm
import torch
from torch import nn
import torch.nn.functional as F


class TimmSeg(nn.Module):
    def __init__(self, encoder: str, in_channels: int, num_classes: int, pretrained=False, dim=64):
        super().__init__()
        self.encoder = timm.create_model(encoder, features_only=True, in_chans=in_channels,
                                         pretrained=pretrained)
        chans = self.encoder.feature_info.channels()
        self.lateral = nn.ModuleList(nn.Conv2d(c, dim, 1) for c in chans)
        self.fuse = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(dim, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = self.encoder(x)
        y = None
        for f, lat in zip(reversed(feats), reversed(self.lateral)):
            f = lat(f)
            y = f if y is None else f + F.interpolate(y, size=f.shape[-2:], mode="bilinear", align_corners=False)
        y = self.head(self.fuse(y))
        return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
