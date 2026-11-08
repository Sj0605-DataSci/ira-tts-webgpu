"""Build the optimised Ira decoder graph for onnxruntime-web (WebGPU) from the shipped 2-bit ONNX model.

What changes versus the shipped graph (details and measurements in bench/README.md):
  * attention: the traced MatMul/Softmax/Concat attention becomes one fused com.microsoft GroupQueryAttention per layer
    (16 query heads, 4 KV heads; rotary is applied inside the op for the sliding-window layers only)
  * KV cache: a static [1, 4, 4096, 128] fp16 buffer per layer, addressed with `seqlens_k` / `total_sequence_length`
    (no per-step Concat)
  * MLP: gate and up are two separate MatMulNBits (no Split); q/k/v are one fused MatMulNBits
  * the 2-bit MatMulNBits weights are kept as shipped (packed blobs reused, not re-quantised)

The result takes T tokens per run (T=1 for decode, T>1 for batched prefill from an empty cache): inputs input_ids [1,T], seqlens_k [1] int32 (= past + T - 1), total_sequence_length int32 scalar (= past + T),
past_key_values.{i}.{key,value}; outputs logits, stop, present.{i}.{key,value}. web/app.js drives it when it detects `seqlens_k`.

Usage:
  python optimize/build_gqa_decoder.py --shipped models/decoder/model.onnx --config config.json --out out/decoder/model.onnx
(config.json is the base model's HF config: rope_theta, layer_norm_eps, layer_types)
"""
import argparse
import json
import re

import numpy as np
import onnx
from onnx import TensorProto as TP, helper as H, numpy_helper as NH


NL, NKV, HD, N_HEADS, HID, INTER, VOCAB_OUT = 36, 4, 128, 16, 2048, 11008, 16385
MAX_POS = 4096  # rotary cache length and static KV buffer length


# ---------------------------------------------------------------- weight extraction
def unpack_nbits(packed, bits, n_rows, n_blocks, block):
    """ORT MatMulNBits packing: values are little-endian within each byte, element j of a byte is bits [j*bits, (j+1)*bits)."""
    per = 8 // bits
    mask = (1 << bits) - 1
    shifts = (np.arange(per) * bits).astype(np.uint8)
    vals = (packed[..., None] >> shifts) & mask                       # [N, nb, blob, per]
    return vals.reshape(n_rows, n_blocks, -1)[..., :block].reshape(n_rows, n_blocks * block)


def unpack_zero_points(zp_packed, bits, n_rows, n_blocks):
    per = 8 // bits
    mask = (1 << bits) - 1
    shifts = (np.arange(per) * bits).astype(np.uint8)
    vals = (zp_packed[..., None] >> shifts) & mask                    # [N, bytes, per]
    return vals.reshape(n_rows, -1)[:, :n_blocks]


def dense_from_nbits(B, scales, zp, bits, block, n, k):
    """Dequantise a MatMulNBits weight to a dense [K, N] float16 matrix so that A @ W == MatMulNBits(A)."""
    nb = k // block
    q = unpack_nbits(B.reshape(n, nb, -1), bits, n, nb, block).astype(np.float32)   # [N, K]
    z = unpack_zero_points(zp.reshape(n, -1), bits, n, nb).astype(np.float32) if zp is not None else np.full((n, nb), 2 ** (bits - 1), np.float32)
    s = scales.reshape(n, nb).astype(np.float32)
    w = (q.reshape(n, nb, block) - z[:, :, None]) * s[:, :, None]
    return np.ascontiguousarray(w.reshape(n, k).T).astype(np.float16)                 # [K, N]



def load_shipped(path):
    return onnx.load(path, load_external_data=True)


def extract_weights(shipped):
    """Return dense fp16 weights keyed by role. Layer weights are found through the node that consumes them, not by name guessing."""
    inits = {i.name: i for i in shipped.graph.initializer}
    W = {}
    for n in shipped.graph.node:
        if n.op_type != "MatMulNBits":
            continue
        a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
        zp = NH.to_array(inits[n.input[3]]) if len(n.input) > 3 and n.input[3] else None
        mm = re.search(r"layers\.(\d+)\.(\w+?)_Q2", n.input[1])
        assert mm, n.input[1]
        i, role = int(mm.group(1)), mm.group(2)
        W[f"layers.{i}.{role}"] = dense_from_nbits(NH.to_array(inits[n.input[1]]), NH.to_array(inits[n.input[2]]), zp,
                                                    a["bits"], a["block_size"], a["N"], a["K"])
    # not quantised: layer-0 down_proj, the logits head, the embedding, the stop head and the layer norms
    for name, t in inits.items():
        m = re.match(r"layers\.(\d+)\.down$", name)
        if m:
            W[f"layers.{m.group(1)}.down"] = NH.to_array(t).astype(np.float16)
    head = [n for n in shipped.graph.node if n.op_type == "MatMul" and n.input[1] in inits and list(inits[n.input[1]].dims) == [HID, VOCAB_OUT]]
    assert len(head) == 1, "logits head not found"
    W["head"] = NH.to_array(inits[head[0].input[1]]).astype(np.float16)
    W["embed"] = NH.to_array(inits["embed"]).astype(np.float16)
    # layer norms: the affine weight is the [2048] initializer consumed by a Mul inside /layers.i/ (final norm has no layer prefix)
    ln = {}
    for n in shipped.graph.node:
        if n.op_type == "Mul":
            for x in n.input:
                if x in inits and list(inits[x].dims) == [HID] and x.startswith("onnx::Mul_"):
                    mm = re.search(r"/layers\.(\d+)/", n.name)
                    key = f"layers.{mm.group(1)}.ln" if mm else "final_norm"
                    if key not in ln:
                        ln[key] = x
    assert len([k for k in ln if k.startswith("layers.")]) == NL and "final_norm" in ln, sorted(ln)
    for k, v in ln.items():
        W[k] = NH.to_array(inits[v]).astype(np.float32)
    W["stop.ln.w"] = NH.to_array(inits["stop.0.weight"]).astype(np.float32)
    W["stop.ln.b"] = NH.to_array(inits["stop.0.bias"]).astype(np.float32)
    W["stop.w1"] = NH.to_array(inits["stop.1.weight"]).T.astype(np.float32)
    W["stop.b1"] = NH.to_array(inits["stop.1.bias"]).astype(np.float32)
    W["stop.w2"] = NH.to_array(inits["stop.3.weight"]).T.astype(np.float32)
    W["stop.b2"] = NH.to_array(inits["stop.3.bias"]).astype(np.float32)
    return W


def rope_config(base_config_path):
    cfg = json.load(open(base_config_path))
    return dict(theta=cfg["rope_theta"], eps=cfg["layer_norm_eps"], types=cfg["layer_types"], logit_scale=cfg.get("logit_scale", 1.0))


# ---------------------------------------------------------------- graph builders

def build_gqa_2bit(shipped, W, rc, out_path):
    """Recommended variant (fused GQA, static KV, gate/up without Split) using the SHIPPED 2-bit MatMulNBits weights,
    so the graph runs on WebGPU exactly like the current model. Packed blobs are concatenated along N for fused q/k/v.
    Layer-0 down_proj, the logits head, embedding and stop head stay dense, as in the shipped model."""
    inits = {i.name: i for i in shipped.graph.initializer}
    nb_attrs = {}
    for n in shipped.graph.node:
        if n.op_type == "MatMulNBits":
            role = re.search(r"layers\.\d+\.(\w+?)_Q2", n.input[1])
            if role and role.group(1) not in nb_attrs:
                nb_attrs[role.group(1)] = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
    nodes, inits_out = [], []
    def init(name, arr): inits_out.append(NH.from_array(np.ascontiguousarray(arr), name)); return name
    def node(op, ins, outs, domain="", **attrs): nodes.append(H.make_node(op, ins, outs, name=outs[0] + "_n", domain=domain, **attrs)); return outs[0]
    def raw(i, roles):
        """packed (B, scales, zero_points) for one or several roles concatenated along the output dimension"""
        parts = {k: [NH.to_array(inits[f"layers.{i}.{r}_{k}"]) for r in roles] for k in ("Q2", "scales", "zero_points")}
        return [np.concatenate(parts[k], axis=0) for k in ("Q2", "scales", "zero_points")]
    def qmm(i, roles, A, out, n_out, attrs_role):
        B, S, Z = raw(i, roles)
        pre = f"layers.{i}.{'_'.join(roles)}"
        a = dict(nb_attrs[attrs_role]); a["N"] = n_out
        return node("MatMulNBits", [A, init(pre + ".B", B), init(pre + ".S", S), init(pre + ".Z", Z)], [out],
                    domain="com.microsoft", **a)

    ax = init("ax1", np.array([1], dtype=np.int64))
    init("embed", W["embed"]); init("head", W["head"]); init("final_norm.w", W["final_norm"])
    inv = 1.0 / (rc["theta"] ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
    ang = np.arange(MAX_POS, dtype=np.float64)[:, None] * inv[None, :]
    init("cos_cache", np.cos(ang).astype(np.float16)); init("sin_cache", np.sin(ang).astype(np.float16))
    ins = [H.make_tensor_value_info("input_ids", TP.INT64, [1, "T"]), H.make_tensor_value_info("seqlens_k", TP.INT32, [1]),
           H.make_tensor_value_info("total_sequence_length", TP.INT32, [])]
    outs = [H.make_tensor_value_info("logits", TP.FLOAT16, [1, VOCAB_OUT]), H.make_tensor_value_info("stop", TP.FLOAT, [1, 1])]
    for i in range(NL):
        for kv in ("key", "value"):
            ins.append(H.make_tensor_value_info(f"past_key_values.{i}.{kv}", TP.FLOAT16, [1, NKV, MAX_POS, HD]))
            outs.append(H.make_tensor_value_info(f"present.{i}.{kv}", TP.FLOAT16, [1, NKV, MAX_POS, HD]))

    x = node("Cast", [node("Gather", ["embed", "input_ids"], ["emb"], axis=0)], ["x_0"], to=TP.FLOAT)
    for i in range(NL):
        p = f"layers.{i}."
        init(p + "ln.w", W[f"layers.{i}.ln"].astype(np.float32))
        h = node("LayerNormalization", [x, p + "ln.w"], [p + "h32"], axis=-1, epsilon=rc["eps"], stash_type=1)
        h = node("Cast", [h], [p + "h"], to=TP.FLOAT16)
        qkv = qmm(i, ["q", "k", "v"], h, p + "qkv_out", HID + 2 * NKV * HD, "q")
        rope = rc["types"][i] == "sliding_attention"
        gqa_in = [qkv, "", "", f"past_key_values.{i}.key", f"past_key_values.{i}.value", "seqlens_k", "total_sequence_length"]
        if rope:
            gqa_in += ["cos_cache", "sin_cache"]
        nodes.append(H.make_node("GroupQueryAttention", gqa_in, [p + "attn", f"present.{i}.key", f"present.{i}.value"], name=p + "gqa",
                                 domain="com.microsoft", num_heads=N_HEADS, kv_num_heads=NKV, do_rotary=int(rope), rotary_interleaved=1,
                                 scale=HD ** -0.5, local_window_size=-1))
        ao = qmm(i, ["o"], p + "attn", p + "o_out", HID, "o")
        g = qmm(i, ["gate"], h, p + "g", INTER, "gate")
        u = qmm(i, ["up"], h, p + "u", INTER, "up")
        act = node("Mul", [node("Mul", [g, node("Sigmoid", [g], [p + "sg"])], [p + "silu"]), u], [p + "act"])
        mo = node("MatMul", [act, p + "down"], [p + "mlp_out"]) if f"layers.{i}.down" in W and i == 0 else qmm(i, ["down"], act, p + "mlp_out", HID, "down")
        if i == 0:
            init(p + "down", W[p + "down"])
        x = node("Add", [node("Add", [x, node("Cast", [ao], [p + "o32"], to=TP.FLOAT)], [p + "r1"]), node("Cast", [mo], [p + "m32"], to=TP.FLOAT)], [f"x_{i + 1}"])

    init("neg1", np.array([-1], dtype=np.int64)); init("big", np.array([2 ** 31 - 1], dtype=np.int64))
    last = node("Slice", [x, "neg1", "big", ax], ["last"])
    hn = node("LayerNormalization", [last, "final_norm.w"], ["hn32"], axis=-1, epsilon=rc["eps"], stash_type=1)
    hn_a = node("Cast", [hn], ["hn"], to=TP.FLOAT16)
    node("Cast", [node("MatMul", [node("Squeeze", [hn_a, ax], ["hn_sq"]), "head"], ["logits_fp16"])], ["logits"], to=TP.FLOAT16)
    init("stop.ln.w", W["stop.ln.w"]); init("stop.ln.b", W["stop.ln.b"]); init("stop.w1", W["stop.w1"]); init("stop.b1", W["stop.b1"])
    init("stop.w2", W["stop.w2"]); init("stop.b2", W["stop.b2"])
    init("c_half", np.array(0.5, dtype=np.float32)); init("c_one", np.array(1.0, dtype=np.float32)); init("c_rs2", np.array(1 / np.sqrt(2), dtype=np.float32))
    hs = node("Squeeze", [hn, ax], ["hs32"])
    st = node("LayerNormalization", [hs, "stop.ln.w", "stop.ln.b"], ["st_ln"], axis=-1, epsilon=1e-5, stash_type=1)
    a1 = node("Add", [node("MatMul", [st, "stop.w1"], ["st_m1"]), "stop.b1"], ["st_a1"])
    erf = node("Erf", [node("Mul", [a1, "c_rs2"], ["st_s"])], ["st_e"])
    gelu = node("Mul", [node("Mul", [a1, node("Add", [erf, "c_one"], ["st_e1"])], ["st_g0"]), "c_half"], ["st_g"])
    node("Sigmoid", [node("Add", [node("MatMul", [gelu, "stop.w2"], ["st_m2"]), "stop.b2"], ["st_a2"])], ["stop"])

    g = H.make_graph(nodes, "ira_decoder", ins, outs, initializer=inits_out)
    model = H.make_model(g, opset_imports=[H.make_opsetid("", 17), H.make_opsetid("com.microsoft", 1)], producer_name="ira-gqa-2bit")
    model.ir_version = 9
    onnx.save_model(model, out_path, save_as_external_data=True, all_tensors_to_one_file=True, location=out_path.split("/")[-1] + ".data", size_threshold=1024)



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shipped", required=True, help="shipped decoder model.onnx (external data file next to it)")
    ap.add_argument("--config", required=True, help="base model config.json")
    ap.add_argument("--out", required=True, help="output model.onnx (weights go to <out>.data)")
    a = ap.parse_args()
    shipped = load_shipped(a.shipped)
    build_gqa_2bit(shipped, extract_weights(shipped), rope_config(a.config), a.out)
    onnx.checker.check_model(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
