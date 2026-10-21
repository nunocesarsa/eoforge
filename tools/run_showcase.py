"""Train one model per downloaded LandCoverNet chip and draw a gallery.

    python tools/run_showcase.py showcase/landcovernet                  # U-Net on every chip
    python tools/run_showcase.py showcase/landcovernet --arch timm:resnet18 --epochs 60
    python tools/run_showcase.py showcase/landcovernet --pooled --holdout 3     # ONE model for all chips

Per-chip mode: each chip gets its own config (<showcase>/configs/) and run (<showcase>/runs/). A model
trained on one 256 x 256 chip has seen one place and one date, so it does not transfer to other chips.
Pooled mode trains one model on all chips and scores it on whole chips it never saw (--holdout), which
is the honest test of whether a model generalises. Results: results.json and gallery.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml

from eoforge.config import load_config
from eoforge.pipeline import train

# LandCoverNet colours (from the dataset documentation); snow/ice is left out (none in these chips)
CLASS_MAP = {1: "water", 2: "artificial bare ground", 3: "natural bare ground",
             5: "woody", 6: "cultivated", 7: "semi-natural vegetation"}
COLOURS = {"water": "#0000ff", "artificial bare ground": "#888888", "natural bare ground": "#d1a46d",
           "woody": "#d64c2b", "cultivated": "#186818", "semi-natural vegetation": "#00ff00"}
CLASSES = list(CLASS_MAP.values())


def config_dict(chip, a) -> dict:
    return {
        "name": chip["chip"],
        "output_dir": "../runs",
        "data": {"source": "local", "path": f"../{chip['image']}"},
        "labels": {"path": f"../{chip['labels']}", "class_map": CLASS_MAP, "classes": CLASSES},
        "model": {"arch": a.arch, "params": json.loads(a.params)},
        "train": {"chip_size": 64, "chips_per_epoch": a.chips, "batch_size": a.batch, "epochs": a.epochs,
                  "lr": a.lr, "val_fraction": 0.25, "block_size": 64,
                  # chips are already sampled class-balanced; weighting the loss as well over-corrects
                  "class_weights": None if a.class_weights == "none" else a.class_weights,
                  "device": a.device, "seed": a.seed},
        "predict": {"tile_size": 256, "overlap": 32, "write_probabilities": False},
    }


def make_config(chip, root: Path, a) -> Path:
    cfg = config_dict(chip, a)
    p = root / "configs" / f"{chip['chip']}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def gallery(root: Path, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap([COLOURS[c] for c in CLASSES])
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n), squeeze=False, constrained_layout=True)
    for row, r in zip(axes, results):
        with rasterio.open(root / r["image"]) as s:
            desc = list(s.descriptions)
            pref = "dry_" if "dry_B04" in desc else ""
            idx = [desc.index(f"{pref}{b}") + 1 if f"{pref}{b}" in desc else i for b, i in (("B04", 3), ("B03", 2), ("B02", 1))]
            rgb = s.read(idx).astype(float)
        valid = rgb.sum(0) > 0
        lo, hi = np.percentile(rgb[:, valid], (2, 98)) if valid.any() else (0, 1)
        rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1).transpose(1, 2, 0)
        lab = rasterio.open(root / r["labels"]).read(1)
        ref = np.full(lab.shape, np.nan)
        for code, nm in CLASS_MAP.items():
            ref[lab == code] = CLASSES.index(nm)
        pdir = root / r.get("pred_dir", f"runs/{r['chip']}/prediction")
        pred = rasterio.open(pdir / "classes.tif").read(1).astype(float)
        pred[pred == 255] = np.nan
        ent = rasterio.open(pdir / "entropy.tif").read(1)
        row[0].imshow(rgb)
        row[0].set_ylabel(r["chip"], fontsize=11)
        row[1].imshow(ref, cmap=cmap, vmin=-.5, vmax=len(CLASSES) - .5, interpolation="nearest")
        row[2].imshow(pred, cmap=cmap, vmin=-.5, vmax=len(CLASSES) - .5, interpolation="nearest")
        row[3].imshow(ent, cmap="magma", vmin=0, vmax=1)
        v = r["heldout"]
        tag = {"unseen": "unseen chip · ", "train": "training chip · "}.get(r.get("role"), "")
        row[2].text(0.02, 0.02, f"{tag}OA {v['overall_accuracy']:.2f} · κ {v['kappa']:.2f}", transform=row[2].transAxes,
                    fontsize=9, va="bottom", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
        for ax in row:
            ax.set_xticks([]); ax.set_yticks([])
    for ax, t in zip(axes[0], ["Sentinel-2 2018, dry season (B4/B3/B2)", "LandCoverNet labels", "Prediction (held-out score)", "Uncertainty (entropy)"]):
        ax.set_title(t, fontsize=11)
    fig.legend(handles=[Patch(color=COLOURS[c], label=c) for c in CLASSES], loc="outside lower center",
               ncol=3, frameon=False)
    fig.savefig(root / "gallery.png", dpi=80, bbox_inches="tight")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("showcase", help="folder written by fetch_landcovernet.py")
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--params", default='{"base": 32, "depth": 3, "dropout": 0.2}', help="JSON model params")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--chips", type=int, default=512, help="training chips per epoch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--class-weights", default="none", choices=["none", "balanced"],
                    help="loss weighting on top of class-balanced chip sampling (default none)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", help="run only these chip ids")
    ap.add_argument("--pooled", action="store_true", help="train one model on all chips instead of one per chip")
    ap.add_argument("--holdout", type=int, default=3, help="pooled mode: whole chips kept out for validation")
    a = ap.parse_args(argv)

    root = Path(a.showcase).resolve()
    if not (root / "manifest.json").exists():
        raise SystemExit(f"No manifest.json in {root}. Download chips first:\n"
                         f"  python tools/fetch_landcovernet.py --n 10 --out {a.showcase}")
    chips = json.loads((root / "manifest.json").read_text())["chips"]
    if a.only:
        chips = [c for c in chips if c["chip"] in set(a.only)]
    if a.pooled:
        return pooled(root, chips, a)
    results = []
    for i, chip in enumerate(chips, 1):
        print(f"\n=== [{i}/{len(chips)}] {chip['chip']} ===", flush=True)
        cfg_path = make_config(chip, root, a)
        train(load_config(cfg_path), config_dir=cfg_path.parent)
        m = json.loads((root / "runs" / chip["chip"] / "metrics.json").read_text())
        v = m["validation_heldout"]
        results.append({**chip, "arch": a.arch, "heldout": {k: v[k] for k in ("n", "overall_accuracy", "kappa", "macro_f1")},
                        "per_class_f1": {c: d["f1"] for c, d in v["per_class"].items() if d["n_ref"] > 0}})
        (root / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'chip':10s} {'px':>6s}  {'OA':>5s}  {'kappa':>5s}  {'mF1':>5s}")
    for r in results:
        v = r["heldout"]
        print(f"{r['chip']:10s} {v['n']:6d}  {v['overall_accuracy']:.3f}  {v['kappa']:.3f}  {v['macro_f1']:.3f}")
    ks = [r["heldout"]["kappa"] for r in results]
    print(f"mean kappa {np.mean(ks):.3f} (min {np.min(ks):.3f}, max {np.max(ks):.3f})")
    gallery(root, results)
    print(f"Gallery: {root / 'gallery.png'}")


def pooled(root: Path, chips, a):
    if len(chips) <= a.holdout:
        raise SystemExit(f"Need more than --holdout {a.holdout} chips")
    rng = np.random.default_rng(a.seed)
    held = sorted(rng.choice([c["chip"] for c in chips], a.holdout, replace=False).tolist())
    cfg = config_dict(chips[0], a)
    cfg["name"] = "pooled"
    cfg["data"] = {"source": "local"}
    cfg["labels"].pop("path")
    cfg["scenes"] = [{"name": c["chip"], "image": f"../{c['image']}", "labels": f"../{c['labels']}"} for c in chips]
    cfg["train"]["val_scenes"] = held
    cfg["train"]["chips_per_epoch"] = a.chips * 2
    p = root / "configs" / "pooled.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"One model on {len(chips) - len(held)} chips; unseen chips for validation: {held}", flush=True)
    train(load_config(p), config_dir=p.parent)
    m = json.loads((root / "runs" / "pooled" / "metrics.json").read_text())
    results = []
    for c in chips:
        ps = m["per_scene"][c["chip"]]
        unseen = c["chip"] in held
        v = ps["validation"] if unseen else ps["training"]
        results.append({**c, "arch": a.arch, "role": "unseen" if unseen else "train",
                        "pred_dir": f"runs/pooled/prediction/{c['chip']}",
                        "heldout": {k: v[k] for k in ("n", "overall_accuracy", "kappa", "macro_f1")}})
    (root / "results_pooled.json").write_text(json.dumps(results, indent=2))
    print(f"\n{'chip':10s} {'role':7s} {'OA':>5s}  {'kappa':>5s}")
    for r in results:
        print(f"{r['chip']:10s} {r['role']:7s} {r['heldout']['overall_accuracy']:.3f}  {r['heldout']['kappa']:.3f}")
    v = m["validation_heldout"]
    print(f"Unseen chips together: OA {v['overall_accuracy']:.3f}, kappa {v['kappa']:.3f}, macro-F1 {v['macro_f1']:.3f}")
    gallery(root, results)
    (root / "gallery.png").replace(root / "gallery_pooled.png")
    print(f"Gallery: {root / 'gallery_pooled.png'}")


if __name__ == "__main__":
    main()
