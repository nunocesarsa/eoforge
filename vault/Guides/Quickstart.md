---
tags: [guide]
---
# Quickstart

## Install
```bash
git clone https://github.com/<you>/eoforge.git && cd eoforge
bash install.sh                  # ./.venv with PyTorch, EOForge, every zoo model and its weights
source .venv/bin/activate
```
Another environment location: `bash install.sh /mnt/d/venvs/eoforge`. Newer GPUs: `TORCH_CUDA=cu128 bash install.sh`. Without the weight download: `SKIP_DOWNLOAD=1`. A plain `pip install -e .` also works if PyTorch is already installed.

## Run ready-made models
```bash
python tools/get_tutorial_data.py         # public sample data, see [[Tutorial]]
eoforge app --folder .                    # http://localhost:8501
```

## Train: the synthetic demo
```bash
eoforge demo --out demo
eoforge train configs/demo_unet.yaml
eoforge predict runs/demo_unet/best.pt demo/scene_other.tif --out pred_other
eoforge export runs/demo_unet/best.pt --out model.onnx
```
See [[Test - Synthetic demo]] for what to expect.

## Train: real Sentinel-2
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/sentinel-hub/eo-learn.git
cd eo-learn && git sparse-checkout set example_data && cd ..
python examples/real_s2_slovenia.py eo-learn/example_data/TestEOPatch real_data
eoforge train configs/real_s2_unet.yaml
```
See [[Test - Real Sentinel-2 Slovenia]]. The tutorial also writes `tutorial/train_worldcover.yaml`: a U-Net on two Sentinel-2 dates with ESA WorldCover as labels.

## Your own data
1. Copy a config, e.g. `configs/real_s2_unet.yaml`.
2. Point `data.path` at a multiband GeoTIFF and `labels.path` at a vector file or a label GeoTIFF (with `labels.class_map`). For several images use `scenes` instead.
3. Set `labels.class_field` and, ideally, `labels.classes` to fix the class order.
4. Choose `train.block_size` so you get at least ~10 blocks with labels (see [[Spatial block validation]]), or hold out whole images with `train.val_scenes`.
5. `eoforge train your_config.yaml`

Outputs land in `runs/<name>/`: see [[train]] and [[write_outputs]].
