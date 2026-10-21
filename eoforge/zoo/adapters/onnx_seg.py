"""Generic ONNX segmentation adapter: the easiest way to add a model you come across.

Card params:
  layout: NCHW | NHWC          input/output layout of the network (default NCHW)
  normalise: {...}             see adapters.base.normalise
  output: logits | probs       (default logits; softmax is applied)
  tile: 512, overlap: 32, batch: 4
  multiple: 1                  pad tiles to a multiple of this (e.g. 32 for some encoders)
  class_values: [..]           output channel -> class code (default 0..K-1)
"""
from __future__ import annotations

import numpy as np

from ..weights import fetch
from .base import Adapter, Result, normalise, register, softmax, tiled_probs


@register("onnx")
class OnnxSegmentation(Adapter):
    requires = ("onnxruntime",)

    def load(self):
        import onnxruntime as ort
        path = fetch(self.card, self.log)["main"]
        avail = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if self.device == "cuda" and "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            try:                      # use the CUDA/cuDNN libraries that come with PyTorch, if present
                ort.preload_dlls()
            except Exception:
                pass
        self.session = ort.InferenceSession(str(path), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        nhwc = self.card.params.get("layout", "NCHW").upper() == "NHWC"
        hw = inp.shape[1:3] if nhwc else inp.shape[2:4]
        self.fixed = int(hw[0]) if all(isinstance(d, int) for d in hw) else None   # fixed input size, if any
        if self.fixed and hw[0] != hw[1]:
            raise ValueError(f"Non-square fixed input {hw} is not supported yet")
        self.log(f"ONNX model {path.name} on {self.session.get_providers()[0]}")

    def predict(self, image, valid, profile, options=None):
        p = {**self.card.params, **(options or {})}
        nhwc = p.get("layout", "NCHW").upper() == "NHWC"
        x = normalise(image, p.get("normalise"))
        x[:, ~valid] = 0

        def run(batch):
            inp = batch.transpose(0, 2, 3, 1) if nhwc else batch
            out = self.session.run(None, {self.input_name: np.ascontiguousarray(inp)})[0]
            if nhwc:
                out = out.transpose(0, 3, 1, 2)
            return out if p.get("output", "logits") == "probs" else softmax(out, 1)

        k = len(self.card.classes) or int(p.get("num_classes", 0))
        if not k:
            s = self.fixed or 64
            k = run(np.zeros((1, x.shape[0], s, s), "float32")).shape[1]
        tile = self.fixed or int(p.get("tile", 512))
        overlap = min(int(p.get("overlap", 32)), tile // 4)
        probs = tiled_probs(run, x, valid, k, tile, overlap, int(p.get("batch", 4)), multiple=int(p.get("multiple", 1)),
                            progress=p.get("_progress"), fixed=bool(self.fixed))
        values = np.asarray(p.get("class_values", [c.value for c in self.card.classes] or list(range(k))))
        classes = values[probs.argmax(0)].astype("uint8")
        return Result(classes=classes, probs=probs)
