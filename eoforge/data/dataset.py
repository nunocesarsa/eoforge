"""Chip sampling with a spatial train/validation split.

Labelled pixels are split by spatial blocks rather than at random, so
validation pixels are not neighbours of training pixels. Random splits of
spatially autocorrelated data inflate accuracy, which matters for EO models.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def block_split(mask: np.ndarray, ignore: int, block: int, val_fraction: float, seed: int):
    """Assign each spatial block that contains labels to train or val.

    Returns (train_mask, val_mask): copies of `mask` with the other split set to ignore.
    """
    h, w = mask.shape
    by, bx = np.arange(h) // block, np.arange(w) // block
    block_id = by[:, None] * (bx.max() + 1) + bx[None, :]
    labelled = mask != ignore
    ids = np.unique(block_id[labelled])
    if ids.size < 2:
        raise ValueError("Labels fall in fewer than 2 spatial blocks; lower train.block_size")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(val_fraction * ids.size)))
    val_ids = ids[:n_val]
    in_val = np.isin(block_id, val_ids)
    train_mask = np.where(in_val, ignore, mask).astype("uint8")
    val_mask = np.where(in_val, mask, ignore).astype("uint8")
    return train_mask, val_mask


class ChipDataset(Dataset):
    """Random chips centred on labelled pixels, from one or several images.

    images: (C, H, W) normalised float32, or a list of them; masks: (H, W) uint8 (or a list) with
    `ignore` where unlabelled. Chip centres are sampled so that every class gets the same share,
    pooled over all images. `radiometric` > 0 jitters each band's gain and offset per chip (in
    normalised units), so the model does not rely on the exact brightness of the training images.
    """

    def __init__(self, image, mask, chip: int, n: int, ignore: int,
                 augment: bool = False, seed: int = 0, fixed: bool = False, radiometric: float = 0.0):
        self.images = image if isinstance(image, (list, tuple)) else [image]
        self.masks = mask if isinstance(mask, (list, tuple)) else [mask]
        self.chip, self.n, self.ignore, self.augment = chip, n, ignore, augment
        self.radiometric = float(radiometric)
        sc, ys, xs, cls = [], [], [], []
        for i, m in enumerate(self.masks):
            y, x = np.nonzero(m != ignore)
            sc.append(np.full(y.size, i)), ys.append(y), xs.append(x), cls.append(m[y, x])
        self.sc, self.ys, self.xs = np.concatenate(sc), np.concatenate(ys), np.concatenate(xs)
        cls = np.concatenate(cls)
        if self.ys.size == 0:
            raise ValueError("No labelled pixels in this split")
        # Sample centres so each class gets a fair share (simple class-balanced sampling).
        weights = np.zeros(cls.size, dtype="float64")
        for c in np.unique(cls):
            sel = cls == c
            weights[sel] = 1.0 / sel.sum()
        self.p = weights / weights.sum()
        self.rng = np.random.default_rng(seed)
        self.fixed_idx = self.rng.choice(self.ys.size, n, p=self.p) if fixed else None

    # single-image access kept for older code
    @property
    def image(self):
        return self.images[0]

    @property
    def mask(self):
        return self.masks[0]

    def __len__(self):
        return self.n

    def _window(self, img, cy, cx):
        _, h, w = img.shape
        half = self.chip // 2
        y0 = int(np.clip(cy - half, 0, max(0, h - self.chip)))
        x0 = int(np.clip(cx - half, 0, max(0, w - self.chip)))
        return y0, x0

    def __getitem__(self, i):
        k = self.fixed_idx[i] if self.fixed_idx is not None else self.rng.choice(self.ys.size, p=self.p)
        image, mask = self.images[self.sc[k]], self.masks[self.sc[k]]
        # jitter so the labelled pixel is not always in the centre
        jitter = self.chip // 4
        cy = self.ys[k] + (self.rng.integers(-jitter, jitter + 1) if self.fixed_idx is None else 0)
        cx = self.xs[k] + (self.rng.integers(-jitter, jitter + 1) if self.fixed_idx is None else 0)
        y0, x0 = self._window(image, cy, cx)
        img = image[:, y0:y0 + self.chip, x0:x0 + self.chip]
        msk = mask[y0:y0 + self.chip, x0:x0 + self.chip]
        img, msk = _pad(img, msk, self.chip, self.ignore)
        if self.augment:
            if self.rng.random() < 0.5:
                img, msk = img[:, :, ::-1], msk[:, ::-1]
            if self.rng.random() < 0.5:
                img, msk = img[:, ::-1, :], msk[::-1, :]
            k90 = int(self.rng.integers(0, 4))
            img, msk = np.rot90(img, k90, axes=(1, 2)), np.rot90(msk, k90)
            if self.radiometric > 0:
                r = self.radiometric
                c = img.shape[0]
                gain = 1 + r * (self.rng.normal() + 0.5 * self.rng.normal(size=c))     # shared + per band
                offset = r * (self.rng.normal() + 0.5 * self.rng.normal(size=c))
                img = img * gain[:, None, None].astype("float32") + offset[:, None, None].astype("float32")
        return torch.from_numpy(np.ascontiguousarray(img, dtype="float32")), \
            torch.from_numpy(np.ascontiguousarray(msk)).long()


def _pad(img, msk, chip, ignore):
    c, h, w = img.shape
    if h == chip and w == chip:
        return img, msk
    pi = np.zeros((c, chip, chip), dtype=img.dtype)
    pm = np.full((chip, chip), ignore, dtype=msk.dtype)
    pi[:, :h, :w], pm[:h, :w] = img, msk
    return pi, pm
