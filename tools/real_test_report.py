"""Score the real Sentinel-2 runs (3 seeds = 3 spatial splits) and draw figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier

from eoforge.config import load_config
from eoforge.data.dataset import block_split
from eoforge.data.raster import rasterise_labels, read_scene
from eoforge.metrics import confusion, summarise

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "docs"
MODELS = [("unet", "U-Net"), ("fcn", "FCN"), ("custom", "Custom (my_model.py)"),
          ("timm", "timm ResNet-18"), ("finetune", "U-Net 2016 → fine-tuned 2017")]
SEEDS = [1, 2, 3]

cfg = load_config(ROOT / "configs/real_s2_unet.yaml")
scene = read_scene(cfg.data.path, cfg.data)
scene16 = read_scene(ROOT / "real_data/s2_slovenia_2016_AprOct.tif", cfg.data)
mask, classes = rasterise_labels(cfg.labels, scene.profile)
k = len(classes)


def read(p):
    with rasterio.open(p) as s:
        return s.read(1)


results = {m: {"oa17": [], "kappa17": [], "f1_17": [], "oa16": [], "kappa16": [], "f1_16": [],
               "ent_ok": [], "ent_bad": [], "f1_class17": []} for m, _ in MODELS + [("rf", "")]}
params = {}
for s in SEEDS:
    tr, va = block_split(mask, 255, cfg.train.block_size, cfg.train.val_fraction, s)
    sel = va != 255
    for m, _ in MODELS:
        run = RUNS / f"real_s2_{m}_s{s}"
        ck = torch.load(run / "best.pt", weights_only=False)
        params[m] = sum(v.numel() for n, v in ck["state_dict"].items() if "running" not in n and "num_batches" not in n)
        if m == "finetune":
            params[m] = (params[m], ck["config"]["model"]["freeze_encoder"])
        p17, p16 = read(run / "prediction/classes.tif"), read(run / "pred_2016/classes.tif")
        e = read(run / "prediction/entropy.tif")
        a, b = summarise(confusion(va, p17, k, 255), classes), summarise(confusion(va, p16, k, 255), classes)
        r = results[m]
        for key, val in [("oa17", a["overall_accuracy"]), ("kappa17", a["kappa"]), ("f1_17", a["macro_f1"]),
                         ("oa16", b["overall_accuracy"]), ("kappa16", b["kappa"]), ("f1_16", b["macro_f1"]),
                         ("ent_ok", float(e[sel & (p17 == va)].mean())), ("ent_bad", float(e[sel & (p17 != va)].mean()))]:
            r[key].append(val)
        r["f1_class17"].append([a["per_class"][c]["f1"] for c in classes])
    # pixel Random Forest baseline on the same split
    X = scene.image.reshape(scene.image.shape[0], -1).T
    lab = tr.ravel() != 255
    rf = RandomForestClassifier(300, class_weight="balanced", n_jobs=2, random_state=s).fit(X[lab], tr.ravel()[lab])
    p17 = rf.predict(X).reshape(tr.shape)
    p16 = rf.predict(scene16.image.reshape(scene16.image.shape[0], -1).T).reshape(tr.shape)
    a, b = summarise(confusion(va, p17, k, 255), classes), summarise(confusion(va, p16, k, 255), classes)
    r = results["rf"]
    for key, val in [("oa17", a["overall_accuracy"]), ("kappa17", a["kappa"]), ("f1_17", a["macro_f1"]),
                     ("oa16", b["overall_accuracy"]), ("kappa16", b["kappa"]), ("f1_16", b["macro_f1"])]:
        r[key].append(val)
    r["f1_class17"].append([a["per_class"][c]["f1"] for c in classes])
    if s == SEEDS[0]:
        np.save(RUNS / "rf_pred_2017_s1.npy", p17)
    results.setdefault("_split", []).append({"seed": s, "val_px_per_class": np.bincount(va[sel], minlength=k).tolist(),
                                             "train_px_per_class": np.bincount(tr[tr != 255], minlength=k).tolist()})

summary = {"classes": classes, "seeds": SEEDS, "params": params, "raw": results, "table": []}
for m, label in MODELS + [("rf", "Random Forest, per pixel (baseline)")]:
    r = results[m]
    row = {"model": label}
    for key in ["oa17", "kappa17", "f1_17", "oa16", "kappa16", "f1_16", "ent_ok", "ent_bad"]:
        if r[key]:
            row[key] = (float(np.mean(r[key])), float(np.std(r[key])))
    f1c = np.array([[np.nan if v is None else v for v in x] for x in r["f1_class17"]], dtype=float)
    row["f1_class17"] = {c: float(np.nanmean(f1c[:, i])) if np.isfinite(f1c[:, i]).any() else None for i, c in enumerate(classes)}
    summary["table"].append(row)
(OUT / "real_s2_results.json").write_text(json.dumps(summary, indent=2))

# ------------------------------------------------------------------ figures (seed 1)
pal = np.array([(27, 120, 55), (166, 217, 106), (217, 166, 48), (200, 50, 60)]) / 255  # forest, grass, shrub, artificial
cmap = ListedColormap(pal)
with rasterio.open(ROOT / "real_data/s2_slovenia_2017_AprOct.tif") as s:
    rgb = s.read([4, 3, 2]).astype(float)
    fc = s.read([8, 4, 3]).astype(float)


def stretch(a):
    lo, hi = np.percentile(a, (2, 98))
    return np.clip((a - lo) / (hi - lo), 0, 1).transpose(1, 2, 0)


ref = mask.astype(float)
ref[mask == 255] = np.nan
_, va1 = block_split(mask, 255, cfg.train.block_size, cfg.train.val_fraction, SEEDS[0])
best = max(MODELS, key=lambda mm: np.mean(results[mm[0]]["kappa17"]))[0]
pred = read(RUNS / f"real_s2_{best}_s1/prediction/classes.tif").astype(float)
ent = read(RUNS / f"real_s2_{best}_s1/prediction/entropy.tif")
pred16 = read(RUNS / f"real_s2_{best}_s1/pred_2016/classes.tif").astype(float)
label_of = dict(MODELS)

fig, ax = plt.subplots(1, 5, figsize=(21, 4.8), constrained_layout=True)
ax[0].imshow(stretch(rgb)); ax[0].set_title("Sentinel-2 2017 Apr–Oct\n(B4/B3/B2)")
ax[1].imshow(ref, cmap=cmap, vmin=-.5, vmax=3.5, interpolation="nearest"); ax[1].set_title("Land-use reference")
yy, xx = np.nonzero(va1 != 255)
ax[1].contour((va1 != 255).astype(float), levels=[0.5], colors="k", linewidths=1.2)
ax[2].imshow(pred, cmap=cmap, vmin=-.5, vmax=3.5, interpolation="nearest"); ax[2].set_title(f"{label_of[best]}, 2017")
ax[3].imshow(pred16, cmap=cmap, vmin=-.5, vmax=3.5, interpolation="nearest"); ax[3].set_title(f"Same model applied to 2016")
im = ax[4].imshow(ent, cmap="magma", vmin=0, vmax=1); ax[4].set_title("Uncertainty 2017 (entropy)")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=pal[i], label=c) for i, c in enumerate(classes)] +
           [Patch(facecolor="none", edgecolor="k", label="held-out blocks (seed 1)")],
           loc="lower center", ncol=5, bbox_to_anchor=(0.42, -0.08), frameon=False)
fig.colorbar(im, ax=ax[4], shrink=.8)
fig.savefig(OUT / "real_s2_maps.png", dpi=95, bbox_inches="tight")

# accuracy chart
labels = [r["model"] for r in summary["table"]]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
for off, key, name, col in [(-0.2, "kappa17", "2017 held-out blocks", "#3b6ea8"), (0.2, "kappa16", "2016 (other year), same blocks", "#d08c2f")]:
    mu = [r[key][0] for r in summary["table"]]
    sd = [r[key][1] for r in summary["table"]]
    ax.bar(x + off, mu, 0.38, yerr=sd, capsize=3, label=name, color=col)
ax.set_xticks(x, [l.replace(" (", "\n(").replace(" → ", "\n→ ") for l in labels], fontsize=8.5)
ax.set_ylabel("Cohen's kappa (mean ± SD, 3 spatial splits)")
ax.set_ylim(0, 1); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "real_s2_kappa.png", dpi=95)

print(f"{'model':40s}  OA17         kappa17      mF1_17       | OA16         kappa16      | H ok / bad")
for r in summary["table"]:
    f = lambda k: f"{r[k][0]:.3f}±{r[k][1]:.3f}" if k in r else "     -     "
    print(f"{r['model']:40s}  {f('oa17')} {f('kappa17')} {f('f1_17')} | {f('oa16')} {f('kappa16')} | "
          f"{r['ent_ok'][0] if 'ent_ok' in r else 0:.2f} / {r['ent_bad'][0] if 'ent_bad' in r else 0:.2f}")
    print("    per-class F1 2017:", {c: (round(v, 2) if v is not None else None) for c, v in r["f1_class17"].items()})
print("splits:", results["_split"])
print("params:", params, "best:", best)
