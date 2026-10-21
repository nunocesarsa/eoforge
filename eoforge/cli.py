"""Command line: eoforge train | predict | export | demo"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eoforge", description="Fine-tune deep learning models on EO imagery.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train/fine-tune from a YAML config, then map the scene")
    t.add_argument("config")

    p = sub.add_parser("predict", help="Apply a trained checkpoint to a GeoTIFF")
    p.add_argument("checkpoint")
    p.add_argument("image")
    p.add_argument("--out", default="prediction")
    p.add_argument("--config", help="optional config for tile size, overlap and outputs")
    p.add_argument("--device", default="auto")

    e = sub.add_parser("export", help="Export a checkpoint to ONNX")
    e.add_argument("checkpoint")
    e.add_argument("--out", default="model.onnx")

    d = sub.add_parser("demo", help="Create a synthetic scene + labels to try the pipeline")
    d.add_argument("--out", default="demo")

    z = sub.add_parser("zoo", help="Ready-made models: list, info, run")
    zs = z.add_subparsers(dest="zcmd", required=True)
    zl = zs.add_parser("list", help="List model cards")
    zl.add_argument("--trained", nargs="*", help="also list best.pt checkpoints under these folders")
    zi = zs.add_parser("info", help="Show a model card")
    zi.add_argument("model")
    zr = zs.add_parser("run", help="Run a model on a GeoTIFF")
    zr.add_argument("model", help="card id (see 'eoforge zoo list') or path to a trained best.pt")
    zr.add_argument("image")
    zr.add_argument("--out", help="output folder (default: <image folder>/eoforge_results)")
    zr.add_argument("--bands", help="1-based band numbers in the model's order, e.g. 2,3,4,8")
    zr.add_argument("--prefix", help="band name prefix for seasonal stacks, e.g. dry")
    zr.add_argument("--device", default="auto")
    zr.add_argument("--probabilities", action="store_true", help="also write probabilities.tif")
    zr.add_argument("--offset", type=float, default=0,
                    help="subtract this from the input (1000 for Sentinel-2 baseline 04.00+ data that keeps the offset)")
    zr.add_argument("--match-resolution", action="store_true", help="resample the image to the model's pixel size")

    zd = zs.add_parser("download", help="Download model weights now (all models, or the ones named)")
    zd.add_argument("models", nargs="*")
    zs.add_parser("status", help="Show which models are ready to run")

    ap_ = sub.add_parser("app", help="Start the web interface (opens in your browser)")
    ap_.add_argument("--port", type=int, default=8501)
    ap_.add_argument("--folder", default=".", help="folder with your images and runs")

    a = ap.parse_args(argv)
    if a.cmd == "train":
        from .pipeline import train
        cfg = load_config(a.config)
        train(cfg, config_dir=Path(a.config).resolve().parent)
    elif a.cmd == "predict":
        from .pipeline import predict
        cfg = load_config(a.config) if a.config else None
        predict(a.checkpoint, a.image, a.out, cfg, a.device, base_dir=Path.cwd())
    elif a.cmd == "export":
        from .pipeline import export_onnx
        export_onnx(a.checkpoint, a.out, base_dir=Path.cwd())
    elif a.cmd == "demo":
        from .demo import make_demo
        make_demo(Path(a.out))
    elif a.cmd == "zoo":
        _zoo(a)
    elif a.cmd == "app":
        import os, subprocess, sys
        app = Path(__file__).parent / "app" / "app.py"
        env = {**os.environ, "EOFORGE_FOLDER": str(Path(a.folder).resolve())}
        port = _free_port(a.port)
        if port != a.port:
            print(f"Port {a.port} is in use (an earlier EOForge app? stop it with: pkill -f 'streamlit run'); using {port}.")
        a.port = port
        print(f"EOForge app: open http://localhost:{a.port} in your browser (Ctrl+C to stop)")
        theme = ["--theme.base", "light", "--theme.primaryColor", "#0E5A63", "--theme.backgroundColor", "#EEF1E8",
                 "--theme.secondaryBackgroundColor", "#E1E7DA", "--theme.textColor", "#1F2A2C"]
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port),
                        "--server.headless", "true", "--browser.gatherUsageStats", "false",
                        "--server.maxUploadSize", "2000", "--client.toolbarMode", "minimal", *theme], env=env)


def _free_port(start: int, tries: int = 20) -> int:
    import socket
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
            try:
                sk.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port between {start} and {start + tries - 1}")


def _zoo(a):
    from . import zoo
    from .zoo.bands import parse_band_list
    if a.zcmd == "list":
        cards = zoo.list_models([Path(p) for p in a.trained] if a.trained is not None else None)
        for c in cards.values():
            print(f"{c.id:22s} {c.name}  [{c.adapter}]  bands {','.join(c.bands) or 'any'}")
        for e in zoo.load_cards.errors:
            print("card error:", e)
    elif a.zcmd == "info":
        import yaml
        c = zoo.get_card(a.model)
        print(yaml.safe_dump({k: v for k, v in c.__dict__.items() if k not in ("path", "classes")}, sort_keys=False))
        print("classes:", ", ".join(f"{x.value}={x.name}" for x in c.classes) or "-")
    elif a.zcmd == "download":
        cards = zoo.load_cards()
        ids = a.models or list(cards)
        failed = []
        for mid in ids:
            print(f"== {mid}")
            try:
                zoo.prefetch(cards[mid], log=lambda m: print("  " + m))
                print("  ready")
            except Exception as e:
                failed.append(mid)
                print(f"  FAILED: {e}")
        if failed:
            raise SystemExit(f"Not downloaded: {', '.join(failed)}")
    elif a.zcmd == "status":
        from .zoo.adapters import ADAPTERS, missing_packages
        for c in zoo.load_cards().values():
            miss = missing_packages(ADAPTERS[c.adapter]) if c.adapter in ADAPTERS else ["adapter " + c.adapter]
            print(f"{c.id:20s} {'ready' if not miss else 'missing ' + ', '.join(miss)}")
    elif a.zcmd == "run":
        if a.model.endswith(".pt"):
            card = zoo.card_from_checkpoint(Path(a.model))
            if card is None:
                raise SystemExit(f"{a.model} is not an eoforge checkpoint")
        else:
            try:
                card = zoo.get_card(a.model)
            except KeyError:
                trained = zoo.trained_cards([Path.cwd()])
                if a.model not in trained:
                    raise
                card = trained[a.model]
        bands = parse_band_list(a.bands) if a.bands else None
        s = zoo.run_model(card, a.image, a.out, bands=bands, prefix=a.prefix, device=a.device,
                          options={"write_probabilities": a.probabilities, "offset": a.offset,
                                   "match_resolution": a.match_resolution})
        for row in s["areas"] if isinstance(s["areas"], list) else []:
            print(f"  {row['name']:22s} {row['percent']:6.2f}%" + (f"  {row['hectares']:10.1f} ha" if "hectares" in row else ""))


if __name__ == "__main__":
    main()
