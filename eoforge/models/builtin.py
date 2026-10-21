"""Small built-in architectures that work for any number of bands."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _block(cin, cout, dropout=0.0):
    layers = [
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, base=32, depth=3, dropout=0.1):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        downs, c = [], in_channels
        for ch in chs[:-1]:
            downs.append(_block(c, ch))
            c = ch
        self.encoder = nn.ModuleDict({
            "downs": nn.ModuleList(downs),
            "bottleneck": _block(chs[-2], chs[-1], dropout),
        })
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.ups.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.dec.append(_block(ch * 2, ch))
        self.head = nn.Conv2d(chs[0], num_classes, 1)
        self.depth = depth

    def forward(self, x):
        h, w = x.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect" if min(h, w) > max(ph, pw) else "constant")
        skips = []
        for down in self.encoder["downs"]:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.encoder["bottleneck"](x)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)[..., :h, :w]


class FCN(nn.Module):
    """Plain stack of 3x3 convs at full resolution: a fast, small baseline."""

    def __init__(self, in_channels, num_classes, width=64, layers=4):
        super().__init__()
        mods, c = [], in_channels
        for _ in range(layers):
            mods += [nn.Conv2d(c, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
            c = width
        self.encoder = nn.Sequential(*mods)
        self.head = nn.Conv2d(width, num_classes, 1)

    def forward(self, x):
        return self.head(self.encoder(x))
