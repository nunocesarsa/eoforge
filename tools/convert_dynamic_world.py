"""Convert Dynamic World (github.com/google/dynamicworld) to ONNX.

    pip install tensorflow-cpu tf2onnx onnx
    python -m tf2onnx.convert --saved-model dynamicworld/model/forward --output raw18.onnx --opset 18
    python tools/convert_dynamic_world.py raw18.onnx dynamic_world_forward.onnx

Replace TF ScaleAndTranslate (antialiased triangle resize, NHWC) with ONNX Resize(linear, antialias=1)."""
import sys, onnx
from onnx import helper, TensorProto
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph
idx = next(i for i, n in enumerate(g.node) if n.op_type == "ScaleAndTranslate")
n = g.node[idx]; x, size, _, _ = n.input; out = n.output[0]; p = n.name
new = [
    helper.make_node("Shape", [x], [p + "/shape"]),
    helper.make_node("Slice", [p + "/shape", p + "/s0", p + "/s1"], [p + "/n"]),
    helper.make_node("Slice", [p + "/shape", p + "/s3", p + "/s4"], [p + "/c"]),
    helper.make_node("Cast", [size], [p + "/hw"], to=TensorProto.INT64),
    helper.make_node("Concat", [p + "/n", p + "/hw", p + "/c"], [p + "/sizes"], axis=0),
    helper.make_node("Resize", [x, "", "", p + "/sizes"], [out], mode="linear", antialias=1,
                     coordinate_transformation_mode="half_pixel"),
]
for nm, v in [("s0", 0), ("s1", 1), ("s3", 3), ("s4", 4)]:
    g.initializer.append(helper.make_tensor(p + "/" + nm, TensorProto.INT64, [1], [v]))
del g.node[idx]
for k, nd in enumerate(new): g.node.insert(idx + k, nd)
used = {i for nd in g.node for i in nd.input}
keep = [t for t in g.initializer if t.name in used]
del g.initializer[:]; g.initializer.extend(keep)
onnx.checker.check_model(m)
onnx.save(m, dst); print("patched ->", dst)
