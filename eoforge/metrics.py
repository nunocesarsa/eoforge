"""Accuracy metrics from a confusion matrix (rows = reference, cols = predicted)."""
from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int, ignore: int) -> np.ndarray:
    sel = y_true != ignore
    t, p = y_true[sel].astype(int), y_pred[sel].astype(int)
    return np.bincount(t * k + p, minlength=k * k).reshape(k, k)


def summarise(cm: np.ndarray, classes: list[str]) -> dict:
    n = cm.sum()
    if n == 0:
        return {"n": 0}
    diag = np.diag(cm).astype(float)
    row, col = cm.sum(1).astype(float), cm.sum(0).astype(float)
    oa = diag.sum() / n
    pe = (row * col).sum() / n ** 2
    kappa = (oa - pe) / (1 - pe) if pe < 1 else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        producer = np.where(row > 0, diag / row, np.nan)   # recall
        user = np.where(col > 0, diag / col, np.nan)       # precision
        f1 = np.where(producer + user > 0, 2 * producer * user / (producer + user), np.nan)
    per_class = {
        c: {"producers_acc": _f(producer[i]), "users_acc": _f(user[i]), "f1": _f(f1[i]), "n_ref": int(row[i])}
        for i, c in enumerate(classes)
    }
    return {
        "n": int(n),
        "overall_accuracy": float(oa),
        "kappa": float(kappa),
        "macro_f1": float(np.nanmean(f1)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _f(x):
    return None if np.isnan(x) else float(x)
