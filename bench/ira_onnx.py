"""Attention variants for the Ira 2-bit decoder, built from the shipped ONNX weights.

Variant summary (A10G, CUDA EP, dense fp16 weights; ms per decode step, mean of 32 steps):

  name                  graph                      KV cache        ORT attention kernel   L=2048 ms  speed-up  top-1 vs shipped
  --------------------  -------------------------  --------------  ---------------------  ---------  --------  ----------------
  shipped_dense         traced MatMul/Softmax      Concat per step explicit ops                21.08      1.00x     1.000 (baseline)
  gqa_concat            fused GroupQueryAttention  Concat per step default (flash)             16.73      1.26x     0.938
  gqa_static            fused GroupQueryAttention  static [1,4,4096,128]  default (flash)       16.14      1.31x     0.938
  gqa_static_no_flash   fused GroupQueryAttention  static                 flash disabled         20.19      1.04x     0.938
  gqa_static_unfused    fused GroupQueryAttention  static                 flash+mem-eff off      not run (no CUDA unfused GQA)

Notes: flash attention drives the gain; static vs concat KV is within noise at these lengths.
Top-1 agreement is over the first 64 steps; the 0.938 mismatch is unverified (fp16 noise suspected).

shipped_dense: the graph exactly as shipped on Hugging Face (traced: MatMul/Softmax attention, Concat KV cache),
               with only the MatMulNBits nodes swapped for the equivalent dense MatMul (CUDA EP has no 2-bit kernel)
gqa_concat:    fused com.microsoft GroupQueryAttention per layer (the build_fast_graph.py design), KV cache concatenated
gqa_static:    same, but the KV cache is a pre-allocated [1,4,MAX,128] buffer shared by past and present (no O(P) copy)

Only the decode step (T=1) is built and benchmarked, which is what the app runs after prefill.
"""
import json
import re
import time

import numpy as np
import onnx
from onnx import TensorProto as TP, helper as H, numpy_helper as NH
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "optimize"))  # on Modal the file sits next to this one
from build_gqa_decoder import (NL, NKV, HD, N_HEADS, HID, INTER, VOCAB_OUT, MAX_POS, unpack_nbits, unpack_zero_points, dense_from_nbits,  # noqa: F401
                               load_shipped, extract_weights, rope_config, build_gqa_2bit)

def check_nbits_roundtrip():
    """Validate the unpacking against ORT's own MatMulNBits kernel (4-bit is the only width ORT CPU/CUDA supports; same code path as 2-bit)."""
    import onnxruntime as ort
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
    rng = np.random.default_rng(0)
    k, n = 256, 96
    W = rng.standard_normal((k, n)).astype(np.float32)
    A = rng.standard_normal((3, k)).astype(np.float32)
    g = H.make_graph([H.make_node("MatMul", ["A", "W"], ["Y"])], "t", [H.make_tensor_value_info("A", TP.FLOAT, [3, k])],
                     [H.make_tensor_value_info("Y", TP.FLOAT, [3, n])], initializer=[NH.from_array(W, "W")])
    m = H.make_model(g, opset_imports=[H.make_opsetid("", 17)])
    q = MatMulNBitsQuantizer(m, block_size=32, is_symmetric=False, algo_config=DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=False, bits=4))
    q.process()
    qm = q.model.model
    node = [x for x in qm.graph.node if x.op_type == "MatMulNBits"][0]
    inits = {i.name: i for i in qm.graph.initializer}
    attrs = {a.name: H.get_attribute_value(a) for a in node.attribute}
    B = NH.to_array(inits[node.input[1]]); S = NH.to_array(inits[node.input[2]])
    Z = NH.to_array(inits[node.input[3]]) if len(node.input) > 3 and node.input[3] else None
    W_dense = dense_from_nbits(B, S, Z, attrs["bits"], attrs["block_size"], attrs["N"], attrs["K"]).astype(np.float32)
    ref = ort.InferenceSession(qm.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {node.input[0]: A})[0]
    ours = A @ W_dense
    err = float(np.abs(ref - ours).max() / (np.abs(ref).max() + 1e-9))
    return err



def build_shipped_dense(shipped, W, out_path):
    """Shipped graph with every MatMulNBits replaced by MatMul against the dense weight. Everything else is untouched."""
    g = shipped.graph
    inits = {i.name: i for i in g.initializer}
    new_inits, nodes = [], []
    for n in g.node:
        if n.op_type == "MatMulNBits":
            wname = n.input[1].replace("_Q2", "") + ".dense"
            role_m = re.search(r"layers\.(\d+)\.(\w+?)_Q2", n.input[1])
            key = f"layers.{role_m.group(1)}.{role_m.group(2)}"
            new_inits.append(NH.from_array(W[key], wname))
            nodes.append(H.make_node("MatMul", [n.input[0], wname], list(n.output), name=n.name))
        else:
            nodes.append(n)
    used = {x for n in nodes for x in n.input}
    keep = [i for i in g.initializer if i.name in used]
    new_graph = H.make_graph(nodes, g.name, list(g.input), list(g.output), initializer=keep + new_inits, value_info=[])
    model = H.make_model(new_graph, opset_imports=list(shipped.opset_import), producer_name="ira-shipped-dense")
    model.ir_version = shipped.ir_version
    onnx.save_model(model, out_path, save_as_external_data=True, all_tensors_to_one_file=True, location=out_path.split("/")[-1] + ".data", size_threshold=1024)


def build_gqa(W, rc, static, out_path, split_gateup=True):
    """Fused GroupQueryAttention decoder. static=True shares one [1,4,MAX_POS,128] KV buffer between past and present."""
    nodes, inits = [], []
    def init(name, arr): inits.append(NH.from_array(np.ascontiguousarray(arr), name)); return name
    def node(op, ins, outs, domain="", **attrs): nodes.append(H.make_node(op, ins, outs, name=outs[0] + "_n", domain=domain, **attrs)); return outs[0]

    inv = 1.0 / (rc["theta"] ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
    ang = np.arange(MAX_POS, dtype=np.float64)[:, None] * inv[None, :]
    init("cos_cache", np.cos(ang).astype(np.float16)); init("sin_cache", np.sin(ang).astype(np.float16))
    init("embed", W["embed"]); init("head", W["head"]); init("final_norm.w", W["final_norm"])
    ins = [H.make_tensor_value_info("input_ids", TP.INT64, [1, 1]), H.make_tensor_value_info("seqlens_k", TP.INT32, [1]),
           H.make_tensor_value_info("total_sequence_length", TP.INT32, [])]
    outs = [H.make_tensor_value_info("logits", TP.FLOAT16, [1, VOCAB_OUT]), H.make_tensor_value_info("stop", TP.FLOAT, [1, 1])]
    past_len = MAX_POS if static else "P"
    for i in range(NL):
        for kv in ("key", "value"):
            ins.append(H.make_tensor_value_info(f"past_key_values.{i}.{kv}", TP.FLOAT16, [1, NKV, past_len, HD]))
            outs.append(H.make_tensor_value_info(f"present.{i}.{kv}", TP.FLOAT16, [1, NKV, past_len if static else "PT", HD]))

    x = node("Gather", ["embed", "input_ids"], ["emb"], axis=0)
    x = node("Cast", [x], ["x_0"], to=TP.FLOAT)
    for i in range(NL):
        p = f"layers.{i}."
        init(p + "ln.w", W[f"layers.{i}.ln"].astype(np.float32))
        init(p + "qkv", np.concatenate([W[p + "q"], W[p + "k"], W[p + "v"]], axis=1))
        init(p + "o", W[p + "o"])
        if split_gateup:
            init(p + "gateup", np.concatenate([W[p + "gate"], W[p + "up"]], axis=1))
            init(p + "split", np.array([INTER, INTER], dtype=np.int64))
        else:
            init(p + "gate", W[p + "gate"]); init(p + "up", W[p + "up"])
        init(p + "down", W[p + "down"])
        h = node("LayerNormalization", [x, p + "ln.w"], [p + "h32"], axis=-1, epsilon=rc["eps"], stash_type=1)
        h = node("Cast", [h], [p + "h"], to=TP.FLOAT16)
        qkv = node("MatMul", [h, p + "qkv"], [p + "qkv_out"])
        rope = rc["types"][i] == "sliding_attention"
        gqa_in = [qkv, "", "", f"past_key_values.{i}.key", f"past_key_values.{i}.value", "seqlens_k", "total_sequence_length"]
        if rope:
            gqa_in += ["cos_cache", "sin_cache"]
        nodes.append(H.make_node("GroupQueryAttention", gqa_in, [p + "attn", f"present.{i}.key", f"present.{i}.value"], name=p + "gqa",
                                 domain="com.microsoft", num_heads=N_HEADS, kv_num_heads=NKV, do_rotary=int(rope), rotary_interleaved=1,
                                 scale=HD ** -0.5, local_window_size=-1))
        ao = node("MatMul", [p + "attn", p + "o"], [p + "o_out"])
        if split_gateup:
            gu = node("MatMul", [h, p + "gateup"], [p + "gu"])
            nodes.append(H.make_node("Split", [gu, p + "split"], [p + "g", p + "u"], name=p + "split_n", axis=-1))
        else:
            node("MatMul", [h, p + "gate"], [p + "g"]); node("MatMul", [h, p + "up"], [p + "u"])
        sg = node("Sigmoid", [p + "g"], [p + "sg"])
        act = node("Mul", [p + "g", sg], [p + "silu"])
        act = node("Mul", [act, p + "u"], [p + "act"])
        mo = node("MatMul", [act, p + "down"], [p + "mlp_out"])
        ao32 = node("Cast", [ao], [p + "o32"], to=TP.FLOAT)
        mo32 = node("Cast", [mo], [p + "m32"], to=TP.FLOAT)
        s1 = node("Add", [x, ao32], [p + "r1"])
        x = node("Add", [s1, mo32], [f"x_{i + 1}"])

    init("neg1", np.array([-1], dtype=np.int64)); init("big", np.array([2 ** 31 - 1], dtype=np.int64)); init("ax1", np.array([1], dtype=np.int64))
    last = node("Slice", [x, "neg1", "big", "ax1"], ["last"])
    hn = node("LayerNormalization", [last, "final_norm.w"], ["hn32"], axis=-1, epsilon=rc["eps"], stash_type=1)
    hn_a = node("Cast", [hn], ["hn"], to=TP.FLOAT16)
    hn2 = node("Squeeze", [hn_a, "ax1"], ["hn_sq"])
    node("MatMul", [hn2, "head"], ["logits_fp16"])
    node("Cast", ["logits_fp16"], ["logits"], to=TP.FLOAT16)
    init("stop.ln.w", W["stop.ln.w"]); init("stop.ln.b", W["stop.ln.b"]); init("stop.w1", W["stop.w1"]); init("stop.b1", W["stop.b1"])
    init("stop.w2", W["stop.w2"]); init("stop.b2", W["stop.b2"])
    init("c_half", np.array(0.5, dtype=np.float32)); init("c_one", np.array(1.0, dtype=np.float32)); init("c_rs2", np.array(1 / np.sqrt(2), dtype=np.float32))
    hs = node("Squeeze", [hn, "ax1"], ["hs32"])
    st = node("LayerNormalization", [hs, "stop.ln.w", "stop.ln.b"], ["st_ln"], axis=-1, epsilon=1e-5, stash_type=1)
    a1 = node("Add", [node("MatMul", [st, "stop.w1"], ["st_m1"]), "stop.b1"], ["st_a1"])
    gelu = node("Mul", [node("Mul", [a1, node("Add", [node("Erf", [node("Mul", [a1, "c_rs2"], ["st_s"])], ["st_e"]), "c_one"], ["st_e1"])], ["st_g0"]), "c_half"], ["st_g"])
    node("Sigmoid", [node("Add", [node("MatMul", [gelu, "stop.w2"], ["st_m2"]), "stop.b2"], ["st_a2"])], ["stop"])

    g = H.make_graph(nodes, "ira_decoder", ins, outs, initializer=inits)
    model = H.make_model(g, opset_imports=[H.make_opsetid("", 17), H.make_opsetid("com.microsoft", 1)], producer_name="ira-bench-gqa")
    model.ir_version = 9
    onnx.save_model(model, out_path, save_as_external_data=True, all_tensors_to_one_file=True, location=out_path.split("/")[-1] + ".data", size_threshold=1024)


# ---------------------------------------------------------------- decode driver
def run_decode(sess, kind, token_ids, dev="cuda"):
    """Decode one token per step with IOBinding (device-resident KV cache). kind in {shipped, concat, static}.
    Yields (step, logits_fp32, stop, wall_ms) for each step."""
    import onnxruntime as ort
    names_kv = [(i, kv) for i in range(NL) for kv in ("key", "value")]
    if kind == "static":
        bufs = {}
        for i, kv in names_kv:
            bufs[(i, kv)] = ort.OrtValue.ortvalue_from_numpy(np.zeros((1, NKV, MAX_POS, HD), np.float16), dev, 0)
    past = {(i, kv): ort.OrtValue.ortvalue_from_numpy(np.zeros((1, NKV, 0, HD), np.float16), dev, 0) for i, kv in names_kv}
    for step, tok in enumerate(token_ids):
        t0 = time.perf_counter()
        io = sess.io_binding()
        io.bind_cpu_input("input_ids", np.array([[tok]], dtype=np.int64))
        if kind == "shipped":
            io.bind_cpu_input("position_ids", np.array([[step]], dtype=np.int64))
        else:
            io.bind_cpu_input("seqlens_k", np.array([step], dtype=np.int32))
            io.bind_cpu_input("total_sequence_length", np.array(step + 1, dtype=np.int32))
        io.bind_output("logits", "cpu")   # bound first so get_outputs() returns [logits, stop, present...] in that order
        io.bind_output("stop", "cpu")
        for i, kv in names_kv:
            io.bind_ortvalue_input(f"past_key_values.{i}.{kv}", bufs[(i, kv)] if kind == "static" else past[(i, kv)])
            if kind == "static":
                io.bind_ortvalue_output(f"present.{i}.{kv}", bufs[(i, kv)])
            else:
                io.bind_output(f"present.{i}.{kv}", dev, 0)
        sess.run_with_iobinding(io)
        outs = io.get_outputs()
        logits = outs[0].numpy().astype(np.float32).reshape(-1)
        stop = float(outs[1].numpy().reshape(-1)[0])
        if kind != "static":
            present = outs[2:]
            past = {k: present[j] for j, k in enumerate(names_kv)}
        wall = (time.perf_counter() - t0) * 1000
        yield step, logits, stop, wall


# ---------------------------------------------------------------- shippable 2-bit build
def dense_twin(path_2bit, out_path):
    """Same graph with every MatMulNBits replaced by MatMul against its dequantised dense weight. Used only to check numerics."""
    m = onnx.load(path_2bit, load_external_data=True)
    inits = {i.name: i for i in m.graph.initializer}
    nodes, extra = [], []
    for n in m.graph.node:
        if n.op_type == "MatMulNBits":
            a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
            zp = NH.to_array(inits[n.input[3]]) if len(n.input) > 3 and n.input[3] else None
            w = dense_from_nbits(NH.to_array(inits[n.input[1]]), NH.to_array(inits[n.input[2]]), zp, a["bits"], a["block_size"], a["N"], a["K"])
            wname = n.input[1] + ".dense"
            extra.append(NH.from_array(w, wname))
            nodes.append(H.make_node("MatMul", [n.input[0], wname], list(n.output), name=n.name))
        else:
            nodes.append(n)
    used = {x for n in nodes for x in n.input}
    keep = [i for i in m.graph.initializer if i.name in used]
    g = H.make_graph(nodes, m.graph.name, list(m.graph.input), list(m.graph.output), initializer=keep + extra)
    model = H.make_model(g, opset_imports=list(m.opset_import), producer_name="ira-dense-twin")
    model.ir_version = m.ir_version
    onnx.save_model(model, out_path, save_as_external_data=True, all_tensors_to_one_file=True, location=out_path.split("/")[-1] + ".data", size_threshold=1024)
