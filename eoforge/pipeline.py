"""End-to-end steps: train (fine-tune) a model and predict a scene."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config, config_to_dict
from .data.dataset import ChipDataset, block_split
from .data.raster import band_stats, normalise, rasterise_labels, read_scene
from .data.sources import get_source
from .metrics import confusion, summarise
from .models.registry import build_model, encoder_parameters
from .predict import predict_scene, write_outputs


def _device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _log(msg: str):
    print(f"[eoforge {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _class_weights(mask, k, ignore, setting):
    if setting is None:
        return None
    if isinstance(setting, list):
        if len(setting) != k:
            raise ValueError(f"train.class_weights has {len(setting)} values for {k} classes")
        return torch.tensor(setting, dtype=torch.float32)
    if setting == "balanced":
        counts = np.bincount(mask[mask != ignore].ravel(), minlength=k).astype(float)
        w = np.where(counts > 0, counts.sum() / (k * np.maximum(counts, 1)), 0.0)
        return torch.tensor(w / w[w > 0].mean(), dtype=torch.float32)
    raise ValueError("train.class_weights must be 'balanced', null or a list")


def _load_weights(model: nn.Module, path: str) -> list[str]:
    """Load matching tensors from a checkpoint; skip ones whose shape differs (e.g. a new head)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    own = model.state_dict()
    loaded, skipped = {}, []
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            loaded[k] = v
        else:
            skipped.append(k)
    model.load_state_dict(loaded, strict=False)
    _log(f"Loaded {len(loaded)}/{len(own)} tensors from {path}; skipped {len(skipped)} (shape/name mismatch)")
    return skipped


def _load_scenes(cfg: Config, run: Path):
    """[(name, Scene, label mask)] for the single image of data.path or every entry of `scenes`."""
    from dataclasses import replace
    if cfg.scenes:
        items = [(sc["name"], sc["image"], sc["labels"]) for sc in cfg.scenes]
    else:
        items = [(cfg.name, str(get_source(cfg.data, run).prepare()), cfg.labels.path)]
    out, classes, names = [], None, None
    for name, img_path, lab_path in items:
        scene = read_scene(img_path, replace(cfg.data, path=str(img_path)))
        mask, cls = rasterise_labels(replace(cfg.labels, path=lab_path), scene.profile)
        mask[~scene.valid] = cfg.labels.ignore_index
        if not (mask != cfg.labels.ignore_index).any():
            raise ValueError(f"Scene '{name}': no labels fall inside the image ({lab_path} vs {img_path})")
        if classes is None:
            classes, names = cls, scene.band_names
        elif cls != classes:
            raise ValueError(f"Scene '{name}' gives classes {cls}, the first scene {classes}. "
                             "Set labels.classes (and labels.class_map for rasters) so all scenes share one list.")
        elif scene.band_names != names:
            _log(f"WARNING scene '{name}' band names {scene.band_names} differ from the first scene {names}; "
                 "check that data.bands selects the same bands in every image.")
        if scene.image.shape[0] != len(names):
            raise ValueError(f"Scene '{name}' has {scene.image.shape[0]} bands, the first scene {len(names)}")
        out.append((name, scene, mask))
    return out, classes


def _pooled_stats(scenes, seed=0, max_pixels=2_000_000):
    """Band mean/std over the valid pixels of all scenes together."""
    rng = np.random.default_rng(seed)
    per = max(1, max_pixels // len(scenes))
    parts = []
    for _, sc, _ in scenes:
        idx = np.flatnonzero(sc.valid.ravel())
        if idx.size > per:
            idx = rng.choice(idx, per, replace=False)
        parts.append(sc.image.reshape(sc.image.shape[0], -1)[:, idx])
    flat = np.concatenate(parts, axis=1)
    mean, std = flat.mean(1), flat.std(1)
    std[std == 0] = 1.0
    return mean.astype("float32"), std.astype("float32")


def train(cfg: Config, config_dir: Path | None = None) -> Path:
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    if cfg.train.num_threads:
        torch.set_num_threads(cfg.train.num_threads)
    device = _device(cfg.train.device)
    run = cfg.run_dir
    run.mkdir(parents=True, exist_ok=True)
    t = cfg.train
    ignore = cfg.labels.ignore_index

    # 1. Data: one or several images, normalised with statistics pooled over all of them
    scenes, classes = _load_scenes(cfg, run)
    k = len(classes)
    mean, std = _pooled_stats(scenes, seed=t.seed)
    images = [normalise(sc.image, sc.valid, mean, std) for _, sc, _ in scenes]
    masks = [m for _, _, m in scenes]
    n_lab = sum(int((m != ignore).sum()) for m in masks)
    _log(f"{len(scenes)} image(s), {images[0].shape[0]} bands; {n_lab:,} labelled px in {k} classes: {classes}")

    # 2. Split: whole held-out scenes (train.val_scenes), or spatial blocks within each scene
    names = [n for n, _, _ in scenes]
    if t.val_scenes:
        unknown = set(t.val_scenes) - set(names)
        if unknown:
            raise ValueError(f"train.val_scenes {sorted(unknown)} not among the scenes {names}")
        tr_masks = [np.full_like(m, ignore) if n in t.val_scenes else m for n, m in zip(names, masks)]
        va_masks = [m if n in t.val_scenes else np.full_like(m, ignore) for n, m in zip(names, masks)]
        split_txt = f"held-out scenes {list(t.val_scenes)}"
    else:
        tr_masks, va_masks = [], []
        for m in masks:
            a, b = block_split(m, ignore, t.block_size, t.val_fraction, t.seed)
            tr_masks.append(a)
            va_masks.append(b)
        split_txt = f"spatial blocks of {t.block_size} px"
    n_tr = sum(int((m != ignore).sum()) for m in tr_masks)
    n_va = sum(int((m != ignore).sum()) for m in va_masks)
    _log(f"Split by {split_txt}: {n_tr:,} train px / {n_va:,} val px")
    tr_keep = [i for i, m in enumerate(tr_masks) if (m != ignore).any()]
    va_keep = [i for i, m in enumerate(va_masks) if (m != ignore).any()]
    train_ds = ChipDataset([images[i] for i in tr_keep], [tr_masks[i] for i in tr_keep], t.chip_size,
                           t.chips_per_epoch, ignore, t.augment, t.seed, radiometric=t.augment_radiometric)
    val_ds = ChipDataset([images[i] for i in va_keep], [va_masks[i] for i in va_keep], t.chip_size,
                         max(64, t.chips_per_epoch // 4), ignore, augment=False, seed=t.seed + 1, fixed=True)
    train_dl = DataLoader(train_ds, batch_size=t.batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=t.batch_size, shuffle=False)

    # 3. Model
    m = cfg.model
    model = build_model(m.arch, images[0].shape[0], k, m.params, base_dir=config_dir)
    if m.weights:
        _load_weights(model, m.weights)
    frozen = encoder_parameters(model) if m.freeze_encoder else []
    for p in frozen:
        p.requires_grad = False
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"Model '{m.arch}': {n_params:,} parameters ({n_train:,} trainable) on {device}")

    weights = _class_weights(np.concatenate([x.ravel() for x in tr_masks]), k, ignore, t.class_weights)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None, ignore_index=ignore)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=t.lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t.lr, total_steps=t.epochs * len(train_dl))

    arch = m.arch
    if ".py:" in arch and config_dir is not None:   # make custom model files findable later
        f, _, fn = arch.rpartition(":")
        if not Path(f).is_absolute():
            arch = f"{(config_dir / f).resolve()}:{fn}"
    meta = {
        "arch": arch, "params": m.params, "classes": classes, "band_names": scenes[0][1].band_names,
        "bands": cfg.data.bands, "mean": mean.tolist(), "std": std.tolist(), "in_channels": int(images[0].shape[0]),
        "ignore_index": ignore, "config": config_to_dict(cfg), "scenes": names,
    }

    # 4. Train
    if t.select_metric not in ("val_loss", "macro_f1", "overall_accuracy", "kappa"):
        raise ValueError("train.select_metric must be val_loss, macro_f1, overall_accuracy or kappa")
    history, best = [], -np.inf
    for epoch in range(1, t.epochs + 1):
        model.train()
        losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        val = _evaluate(model, val_dl, k, ignore, classes, device, loss_fn)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val["loss"],
                        "val_oa": val["overall_accuracy"], "val_macro_f1": val["macro_f1"]})
        _log(f"epoch {epoch:3d}/{t.epochs}  loss {np.mean(losses):.4f}  val loss {val['loss']:.4f}  "
             f"val OA {val['overall_accuracy']:.3f}  macro-F1 {val['macro_f1']:.3f}")
        score = -val["loss"] if t.select_metric == "val_loss" else val[t.select_metric]
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, **meta}, run / "best.pt")
    torch.save({"state_dict": model.state_dict(), "epoch": t.epochs, **meta}, run / "last.pt")

    # 5. Honest final check: map every scene with the best model and score every held-out pixel
    ckpt = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    cm_va, cm_tr = np.zeros((k, k), np.int64), np.zeros((k, k), np.int64)
    per_scene = {}
    for i, (name, sc, _) in enumerate(scenes):
        probs = predict_scene(model, images[i], sc.valid, k, cfg.predict.tile_size, cfg.predict.overlap,
                              cfg.predict.batch_size, device)
        pred = probs.argmax(0)
        cv, ct = confusion(va_masks[i], pred, k, ignore), confusion(tr_masks[i], pred, k, ignore)
        cm_va += cv
        cm_tr += ct
        if len(scenes) > 1:
            per_scene[name] = {"validation": summarise(cv, classes) if cv.sum() else None,
                               "training": summarise(ct, classes) if ct.sum() else None}
        out = run / "prediction" if len(scenes) == 1 else run / "prediction" / name
        paths = write_outputs(out, sc.profile, probs, sc.valid, classes,
                              cfg.predict.write_entropy, cfg.predict.write_probabilities)
    report = {
        "best_epoch": ckpt["epoch"],
        "split": split_txt,
        "validation_heldout": summarise(cm_va, classes),
        "training_pixels": summarise(cm_tr, classes),
        **({"per_scene": per_scene} if per_scene else {}),
        "history": history,
    }
    report["validation_heldout_blocks"] = report["validation_heldout"]       # older name, kept for tools
    (run / "metrics.json").write_text(json.dumps(report, indent=2))
    v = report["validation_heldout"]
    _log(f"Held-out ({split_txt}, best epoch {ckpt['epoch']}): OA {v['overall_accuracy']:.3f}, "
         f"kappa {v['kappa']:.3f}, macro-F1 {v['macro_f1']:.3f} on {v['n']:,} px")
    _log(f"Maps written to {run / 'prediction'}")
    return run / "best.pt"


@torch.no_grad()
def _evaluate(model, dl, k, ignore, classes, device, loss_fn=None):
    model.eval()
    cm = np.zeros((k, k), dtype=np.int64)
    losses = []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if loss_fn is not None:
            losses.append(loss_fn(logits, y).item())
        cm += confusion(y.cpu().numpy(), logits.argmax(1).cpu().numpy(), k, ignore)
    out = summarise(cm, classes)
    out["loss"] = float(np.mean(losses)) if losses else float("nan")
    return out


def load_model(checkpoint: str | Path, device="cpu", base_dir: Path | None = None):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], ckpt["in_channels"], len(ckpt["classes"]), ckpt["params"], base_dir)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def predict(checkpoint: str | Path, image_path: str | Path, out_dir: str | Path,
            cfg: Config | None = None, device="auto", base_dir: Path | None = None) -> dict:
    """Apply a trained checkpoint to any GeoTIFF with the same bands."""
    from .config import DataConfig, PredictConfig
    device = _device(device)
    model, ckpt = load_model(checkpoint, device, base_dir)
    pc = cfg.predict if cfg else PredictConfig()
    bands = ckpt.get("bands")
    import rasterio
    with rasterio.open(image_path) as src:
        desc = [d or "" for d in src.descriptions]
    want = [str(n) for n in ckpt.get("band_names", [])]
    if want and all(n in desc for n in want):
        by_name = [desc.index(n) + 1 for n in want]
        if bands and by_name != list(bands):
            _log(f"Bands picked by name {want} -> {by_name} (the training image had them at {bands})")
        bands = by_name
    dc = DataConfig(path=str(image_path), bands=bands)
    scene = read_scene(image_path, dc)
    if scene.image.shape[0] != ckpt["in_channels"]:
        raise ValueError(f"Image has {scene.image.shape[0]} bands; model expects {ckpt['in_channels']} "
                         f"({ckpt['band_names']})")
    mean, std = np.array(ckpt["mean"], "float32"), np.array(ckpt["std"], "float32")
    check_band_shift(scene, mean, std, ckpt["band_names"])
    image = normalise(scene.image, scene.valid, mean, std)
    probs = predict_scene(model, image, scene.valid, len(ckpt["classes"]), pc.tile_size, pc.overlap,
                          pc.batch_size, device)
    paths = write_outputs(Path(out_dir), scene.profile, probs, scene.valid, ckpt["classes"],
                          pc.write_entropy, pc.write_probabilities)
    _log(f"Wrote {', '.join(paths.values())}")
    return paths


def check_band_shift(scene, mean, std, band_names, threshold=2.0) -> dict:
    """Warn when a new image's band means sit far from the training statistics.

    The shift is measured in training standard deviations. Large shifts (e.g. an
    atmospheric band such as B10 between years, or a different sensor/processing
    level) are a common reason a model transfers badly; consider dropping the band.
    """
    new_mean, _ = band_stats(scene)
    shift = (new_mean - mean) / std
    out = {b: float(s) for b, s in zip(band_names, shift)}
    flagged = {b: s for b, s in out.items() if abs(s) > threshold}
    if flagged:
        _log("WARNING band shift vs training data (in training SDs): "
             + ", ".join(f"{b} {s:+.1f}" for b, s in flagged.items())
             + ". Predictions may be unreliable; consider excluding these bands (data.bands).")
    return out


def export_onnx(checkpoint: str | Path, out_path: str | Path, base_dir: Path | None = None, tile=256) -> Path:
    """Export to ONNX for offline/field use (e.g. onnxruntime, browser). Needs `pip install onnx`."""
    model, ckpt = load_model(checkpoint, "cpu", base_dir)
    x = torch.zeros(1, ckpt["in_channels"], tile, tile)
    out_path = Path(out_path)
    torch.onnx.export(model, x, out_path, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    sidecar = {k: ckpt[k] for k in ("classes", "band_names", "mean", "std", "in_channels", "arch")}
    sidecar["note"] = "Normalise inputs as (x - mean) / std per band; apply softmax to logits."
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    _log(f"Exported {out_path} (+ {out_path.with_suffix('.json').name})")
    return out_path
