"""Modal harness: profile and benchmark attention variants of the Ira 2-bit decoder on a GPU.

    modal run bench/modal_app.py::validate              # dequant round-trip + graph sanity (cheap)
    modal run bench/modal_app.py                        # all variants in parallel, writes bench/results/

Each variant runs in its own A10G container so CUDA state and memory never leak between variants.
Weights come from the HF model repo and are cached in the Modal volume `ira-tts-weights`.
"""
import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

HF_REPO = "Sanyam0605/ira-tts-2bit-web"
BASE_CONFIG_URL = "https://huggingface.co/rumik-ai/rumik-oss-1/resolve/main/config.json"
VOL = modal.Volume.from_name("ira-tts-weights", create_if_missing=True)
WEIGHTS = "/weights"
GPU = "A10G"
STEPS = 2080                    # decode steps per variant (context grows 0 -> 2080)
CHECKPOINTS = [64, 512, 1024, 2048]
WINDOW = 32                     # mean over steps [L, L+WINDOW)
COMPARE_STEPS = 64              # logits kept for the correctness check

image_base = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .pip_install("onnxruntime-gpu==1.23.2", "onnx==1.17.0", "numpy<2", "huggingface_hub==0.30.2")
)
image = image_base.add_local_file(HERE / "ira_onnx.py", "/root/ira_onnx.py").add_local_file(HERE.parent / "optimize" / "build_gqa_decoder.py", "/root/build_gqa_decoder.py")
app = modal.App("ira-tts-attention-bench", image=image)

# Variant table (numbers from results/bench.json; see ira_onnx.py docstring for the full table):
#   name                 | graph kind | ORT env overrides                                  | L=2048 ms | speed-up
#   shipped_dense        | shipped    | none (explicit attention, Concat KV)               | 21.08     | 1.00x
#   gqa_concat           | concat     | none (fused GQA, Concat KV)                        | 16.73     | 1.26x
#   gqa_static           | static     | none (fused GQA, static KV buffer)                 | 16.14     | 1.31x
#   gqa_static_no_flash  | static     | ORT_DISABLE_FLASH_ATTENTION=1                      | 20.19     | 1.04x
#   gqa_static_unfused   | static     | flash + mem-efficient disabled (not run: no kernel)| n/a       | n/a
VARIANTS = {
    # name: (graph kind, env overrides for ORT's fused-attention kernel selection)
    "shipped_dense": ("shipped", {}),
    "gqa_concat": ("concat", {}),
    "gqa_static": ("static", {}),
    "gqa_static_no_flash": ("static", {"ORT_DISABLE_FLASH_ATTENTION": "1"}),
    "gqa_static_unfused": ("static", {"ORT_DISABLE_FLASH_ATTENTION": "1", "ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION": "1"}),
    # round 2: launch-overhead and projection-split variants
    "gqa_static_nosplit": ("static_nosplit", {}),                  # gate/up as two MatMuls, no Split
    "gqa_static_cudagraph": ("static", {"CUDA_GRAPH": "1"}),       # static KV decode step captured as a CUDA graph
    "gqa_static_nosplit_cudagraph": ("static_nosplit", {"CUDA_GRAPH": "1"}),
}


@app.function(volumes={WEIGHTS: VOL}, timeout=1800)
def fetch_weights():
    from huggingface_hub import snapshot_download, hf_hub_download
    import urllib.request
    p = snapshot_download(HF_REPO, local_dir=f"{WEIGHTS}/ira", allow_patterns=["decoder/*", "assets/*"])
    urllib.request.urlretrieve(BASE_CONFIG_URL, f"{WEIGHTS}/rumik_oss_1_config.json")
    VOL.commit()
    return p


def _prepare(kind_graph, workdir):
    """Build the requested graph into workdir (container-local disk). Returns (path, kind)."""
    import onnx
    import ira_onnx as I
    shipped = I.load_shipped(f"{WEIGHTS}/ira/decoder/model.onnx")
    W = I.extract_weights(shipped)
    rc = I.rope_config(f"{WEIGHTS}/rumik_oss_1_config.json")
    os.makedirs(workdir, exist_ok=True)
    out = f"{workdir}/model.onnx"
    if kind_graph == "shipped":
        I.build_shipped_dense(shipped, W, out)
    else:
        I.build_gqa(W, rc, static=kind_graph.startswith("static"), out_path=out,
                    split_gateup=(kind_graph != "static_nosplit"))
    return out


def _session(path, env):
    env = dict(env)
    graph_capture = env.pop("CUDA_GRAPH", None) == "1"
    for k, v in env.items():
        os.environ[k] = v
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cuda_opts = {"device_id": 0, "enable_cuda_graph": "1" if graph_capture else "0"}
    sess = ort.InferenceSession(path, so, providers=[("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"])
    assert sess.get_providers()[0] == "CUDAExecutionProvider", sess.get_providers()
    return sess


@app.function(volumes={WEIGHTS: VOL}, gpu=GPU, memory=32768, timeout=1800)
def validate():
    import numpy as np
    import ira_onnx as I
    import onnxruntime as ort
    rel = I.check_nbits_roundtrip()
    print(f"[validate] MatMulNBits 4-bit round-trip rel. error vs ORT kernel: {rel:.3e}")
    print(f"[validate] ORT {ort.__version__} providers: {ort.get_available_providers()}")
    assert rel < 1e-2, "dequantisation disagrees with ORT"
    shipped = I.load_shipped(f"{WEIGHTS}/ira/decoder/model.onnx")
    W = I.extract_weights(shipped)
    print("[validate] layer-0 shapes:", {k: tuple(W[f'layers.0.{k}'].shape) for k in ["q", "k", "v", "o", "gate", "up"]},
          "| head", W["head"].shape, "| embed", W["embed"].shape, "| final_norm", W["final_norm"].shape)
    # per-channel sanity: 2-bit dequantised weights take at most 4 distinct values per block
    w = W["layers.10.q"][:32, :].astype(np.float32)          # K rows 0..31 = first quantisation block, for every output column
    print("[validate] layer-10 q, first 2-bit block: max distinct levels per output column:", max(len(np.unique(w[:, n])) for n in range(w.shape[1])))
    print("[validate] layer-0 down (fp16 in shipped graph) rms:", float(np.sqrt((W['layers.0.down'].astype(np.float32) ** 2).mean())))
    return {"nbits_rel_err": rel}


@app.function(volumes={WEIGHTS: VOL}, gpu=GPU, memory=32768, timeout=3600)
def bench_variant(name: str):
    try:
        return _bench_variant(name)
    except Exception as e:  # keep the other variants alive; record why this one could not run
        import traceback
        print(f"[{name}] FAILED: {e}", flush=True)
        return {"name": name, "kind": VARIANTS[name][0], "env": VARIANTS[name][1], "error": f"{type(e).__name__}: {str(e)[:400]}"}


def _bench_variant(name: str):
    import time
    import numpy as np
    import ira_onnx as I
    kind, env = VARIANTS[name]
    t0 = time.time()
    path = _prepare(kind, f"/tmp/work/{name}")
    t_build = time.time() - t0
    sess = _session(path, env)
    rng = np.random.default_rng(1234)
    tokens = rng.integers(16, 32016, size=STEPS).tolist()   # decode-only, token ids in the text range (values do not change cost)
    steps_ms, keep_logits, stops = [], [], []
    t_run = time.time()
    decode_kind = "static" if kind.startswith("static") else kind   # nosplit is still a static-KV graph
    for step, logits, stop, wall in I.run_decode(sess, decode_kind, tokens):
        steps_ms.append(wall)
        stops.append(stop)
        if step < COMPARE_STEPS:
            keep_logits.append(logits.astype(np.float16))
        if step % 256 == 0:
            print(f"[{name}] step {step} {wall:.2f} ms", flush=True)
    run_s = time.time() - t_run
    ms = np.array(steps_ms)
    lat = {L: float(ms[L:L + WINDOW].mean()) for L in CHECKPOINTS}
    lat_p50 = {L: float(np.median(ms[L:L + WINDOW])) for L in CHECKPOINTS}
    print(f"[{name}] build {t_build:.0f}s run {run_s:.0f}s | " + " ".join(f"L{L}={lat[L]:.2f}ms" for L in CHECKPOINTS), flush=True)
    return {
        "name": name, "kind": kind, "env": env, "build_s": t_build, "run_s": run_s,
        "lat_mean_ms": lat, "lat_median_ms": lat_p50,
        "ms_per_step": steps_ms, "stop": stops,
        "logits_fp16": np.stack(keep_logits).tobytes().hex(),
        "providers": sess.get_providers(),
    }


@app.function(volumes={WEIGHTS: VOL}, gpu=GPU, memory=32768, timeout=3600)
def profile_variant(name: str, steps: int = 512):
    """ORT per-operator profile over the first `steps` decode steps. CUDA kernel time is host-observed, so read it as a relative split."""
    import glob
    import collections
    import numpy as np
    import onnxruntime as ort
    import ira_onnx as I
    kind, env = VARIANTS[name]
    path = _prepare(kind, f"/tmp/work/{name}")
    for k, v in env.items():
        os.environ[k] = v
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_profiling = True
    so.profile_file_prefix = f"/tmp/prof_{name}"
    sess = ort.InferenceSession(path, so, providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
    rng = np.random.default_rng(1234)
    tokens = rng.integers(16, 32016, size=steps).tolist()
    for _ in I.run_decode(sess, "static" if kind.startswith("static") else kind, tokens):
        pass
    fn = sess.end_profiling()
    events = json.load(open(fn))
    by_op = collections.Counter()
    by_node = collections.Counter()
    n_ev = 0
    for e in events:
        if e.get("cat") != "Node" or not e.get("name", "").endswith("kernel_time"):
            continue
        args = e.get("args", {})
        if args.get("provider") != "CUDAExecutionProvider":
            continue
        n_ev += 1
        by_op[args.get("op_name", "?")] += e["dur"]
        by_node[e["name"].replace("_kernel_time", "")] += e["dur"]
    total = sum(by_op.values())
    print(f"[profile {name}] {n_ev} CUDA node events, total {total/1e3:.0f} ms over {steps} steps")
    top_ops = [(k, v / 1e3, v / total) for k, v in by_op.most_common(20)]
    attn_keys = ("gqa", "attn", "softmax", "concat", "kv", "present", "past")
    attn_time = sum(v for k, v in by_node.items() if any(a in k.lower() for a in attn_keys))
    return {"name": name, "steps": steps, "total_ms": total / 1e3, "top_ops": top_ops,
            "attn_share_by_name": attn_time / total if total else None}


@app.function(timeout=600)
def compare_and_table(results: list):
    """Runs in the container (numpy lives there): correctness vs the shipped graph and the markdown table."""
    import numpy as np
    lines_err = [f"| {r['name']} | not run: {r['error']} |" for r in results if "error" in r]
    results = [r for r in results if "error" not in r]
    base = next(r for r in results if r["name"] == "shipped_dense")
    base_logits = np.frombuffer(bytes.fromhex(base["logits_fp16"]), np.float16).reshape(COMPARE_STEPS, -1).astype(np.float32)
    lines = ["| variant | build s | L=64 ms | L=512 ms | L=1024 ms | L=2048 ms | speed-up @2048 | top-1 agree | max abs logit diff |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lg = np.frombuffer(bytes.fromhex(r["logits_fp16"]), np.float16).reshape(COMPARE_STEPS, -1).astype(np.float32)
        d = np.abs(lg - base_logits)
        r["check"] = {"max_abs_logit_diff": float(d.max()), "mean_abs_logit_diff": float(d.mean()),
                      "top1_agree": float((lg.argmax(-1) == base_logits.argmax(-1)).mean()),
                      "stop_max_abs_diff": float(np.abs(np.array(r["stop"][:COMPARE_STEPS]) - np.array(base["stop"][:COMPARE_STEPS])).max())}
        lm = r["lat_mean_ms"]
        sp = base["lat_mean_ms"][2048] / lm[2048]
        lines.append(f"| {r['name']} | {r['build_s']:.0f} | {lm[64]:.2f} | {lm[512]:.2f} | {lm[1024]:.2f} | {lm[2048]:.2f} | {sp:.2f}x | "
                     f"{r['check']['top1_agree']:.3f} | {r['check']['max_abs_logit_diff']:.3f} |")
    lines += lines_err
    slim = [{k: v for k, v in r.items() if k != "logits_fp16"} for r in results]
    return {"table": "\n".join(lines), "rows": slim}


@app.local_entrypoint()
def main(only: str = "", profile: bool = True):
    names = [n for n in VARIANTS if not only or n in only.split(",")]
    print("building + benchmarking:", names, flush=True)
    results = list(bench_variant.map(names))
    summary = compare_and_table.remote(results)
    print("\n" + summary["table"])
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "bench.json").write_text(json.dumps(summary["rows"], indent=1, default=str))
    if profile:
        profs = list(profile_variant.map(["shipped_dense", "gqa_static"]))
        (out / "profile.json").write_text(json.dumps(profs, indent=1, default=str))
        for p in profs:
            print(f"\n{p['name']}: total {p['total_ms']:.0f} ms over {p['steps']} steps, attention-named share {p['attn_share_by_name']}")
            for k, ms, frac in p["top_ops"][:10]:
                print(f"  {k:28s} {ms:10.1f} ms  {frac*100:5.1f}%")
    print(f"\nwrote {out}/bench.json")


@app.function(volumes={WEIGHTS: VOL}, gpu=GPU, memory=32768, timeout=3600)
def verify_2bit(steps: int = 64):
    """Build the shippable 2-bit graph, check it, and check its numerics via a dense twin against the benchmarked nosplit graph."""
    import numpy as np
    import onnx
    import ira_onnx as I
    shipped = I.load_shipped(f"{WEIGHTS}/ira/decoder/model.onnx")
    W = I.extract_weights(shipped)
    rc = I.rope_config(f"{WEIGHTS}/rumik_oss_1_config.json")
    os.makedirs("/tmp/work/ship", exist_ok=True)
    p2 = "/tmp/work/ship/model.onnx"
    I.build_gqa_2bit(shipped, W, rc, p2)
    onnx.checker.check_model(p2, full_check=False)
    print("[verify] 2-bit graph passes onnx.checker", flush=True)
    onnx_model = onnx.load(p2, load_external_data=False)
    from collections import Counter
    print("[verify] ops:", Counter(n.op_type for n in onnx_model.graph.node).most_common(12), flush=True)
    twin = "/tmp/work/twin/model.onnx"
    os.makedirs("/tmp/work/twin", exist_ok=True)
    I.dense_twin(p2, twin)
    ref = "/tmp/work/ref_nosplit/model.onnx"
    os.makedirs("/tmp/work/ref_nosplit", exist_ok=True)
    I.build_gqa(W, rc, static=True, out_path=ref, split_gateup=False)
    rng = np.random.default_rng(1234)
    tokens = rng.integers(16, 32016, size=steps).tolist()
    outs = {}
    for name, path in [("twin", twin), ("ref", ref)]:
        sess = _session(path, {})
        outs[name] = np.stack([lg for _, lg, _, _ in I.run_decode(sess, "static", tokens)])
        del sess
    d = np.abs(outs["twin"] - outs["ref"])
    top1 = float((outs["twin"].argmax(-1) == outs["ref"].argmax(-1)).mean())
    print(f"[verify] twin vs benchmarked nosplit graph over {steps} steps: max|dlogit|={d.max():.4g} top1={top1:.3f}", flush=True)
    return {"max_abs_diff": float(d.max()), "top1": top1, "steps": steps}


@app.local_entrypoint()
def verify():
    print(verify_2bit.remote())


image_torch = (image_base.pip_install("torch==2.5.1")
               .add_local_file(HERE / "ira_onnx.py", "/root/ira_onnx.py")
               .add_local_file(HERE.parent / "pipeline" / "web_tts" / "ref_decoder.py", "/root/ref_decoder.py"))


@app.function(image=image_torch, volumes={WEIGHTS: VOL}, gpu=GPU, memory=65536, timeout=3600)
def accuracy_check(steps: int = 64):
    """fp32 PyTorch reference (ref_decoder.py, same dequantised weights) vs the fp16 shipped graph and the fp16 fused nosplit graph."""
    import gc
    import numpy as np
    import torch
    import ira_onnx as I
    from ref_decoder import WebDecoder, empty_past
    shipped = I.load_shipped(f"{WEIGHTS}/ira/decoder/model.onnx")
    W = I.extract_weights(shipped)
    del shipped
    cfg = json.load(open(f"{WEIGHTS}/rumik_oss_1_config.json"))
    cfg.setdefault("logit_scale", 1.0)
    rng = np.random.default_rng(1234)
    tokens = rng.integers(16, 32016, size=STEPS).tolist()[:steps]

    m = WebDecoder(cfg, 48400)
    f = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    with torch.no_grad():
        m.embed.copy_(f(W["embed"])); m.out_rows.copy_(f(W["head"]).t()); m.norm.weight.copy_(f(W["final_norm"]))
        for i, L in enumerate(m.layers):
            L.ln.weight.copy_(f(W[f"layers.{i}.ln"]))
            for r in ("q", "k", "v", "o", "gate", "up", "down"):
                getattr(L, r).copy_(f(W[f"layers.{i}.{r}"]))
        m.stop[0].weight.copy_(f(W["stop.ln.w"])); m.stop[0].bias.copy_(f(W["stop.ln.b"]))
        m.stop[1].weight.copy_(f(W["stop.w1"]).t()); m.stop[1].bias.copy_(f(W["stop.b1"]))
        m.stop[3].weight.copy_(f(W["stop.w2"]).t()); m.stop[3].bias.copy_(f(W["stop.b2"]))
    m = m.eval()   # CPU: ref_decoder builds its causal mask on CPU
    past = list(empty_past(cfg, torch.float32))
    ref = []
    with torch.no_grad():
        for t, tok in enumerate(tokens):
            out = m(torch.tensor([[tok]]), torch.tensor([[t]]), *past)
            past = list(out[2:])
            ref.append(out[0][0].float().cpu().numpy())
    ref = np.stack(ref)
    del m, past
    gc.collect()
    print(f"[acc] fp32 reference done ({steps} steps)", flush=True)

    res = {}
    outs = {}
    for name in ("shipped_dense", "gqa_static_nosplit"):
        kind, env = VARIANTS[name]
        sess = _session(_prepare(kind, f"/tmp/work/{name}"), env)
        dk = "static" if kind.startswith("static") else kind
        outs[name] = np.stack([lg for _, lg, _, _ in I.run_decode(sess, dk, tokens)])
        del sess
        gc.collect()
        o = outs[name]
        d = np.abs(o - ref)
        mism = np.where(o.argmax(-1) != ref.argmax(-1))[0]
        gaps = [float(ref[t, ref[t].argmax()] - ref[t, o[t].argmax()]) for t in mism]   # how far below ref's best the variant's pick is, in fp32 logits
        res[name] = {"top1_vs_fp32": float((o.argmax(-1) == ref.argmax(-1)).mean()), "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                     "mismatch_steps": mism.tolist(), "ref_logit_gap_at_mismatch": gaps,
                     "ref_top1_margin_median": float(np.median(np.sort(ref, -1)[:, -1] - np.sort(ref, -1)[:, -2]))}
        print(f"[acc] {name}: top1 vs fp32 {res[name]['top1_vs_fp32']:.3f} max|d| {res[name]['max_abs']:.4f} mean|d| {res[name]['mean_abs']:.5f} "
              f"mismatches {len(mism)} gaps {['%.3f' % g for g in gaps]}", flush=True)
    a, b = outs["shipped_dense"], outs["gqa_static_nosplit"]
    res["fp16_variants_top1_agree"] = float((a.argmax(-1) == b.argmax(-1)).mean())
    print("[acc] shipped_fp16 vs fused_fp16 top1 agree:", res["fp16_variants_top1_agree"], flush=True)
    return res


@app.local_entrypoint()
def accuracy():
    r = accuracy_check.remote()
    (HERE / "results" / "accuracy_fp32.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


@app.function(volumes={WEIGHTS: VOL}, memory=32768, timeout=3600)
def build_ship():
    """Build the shippable 2-bit fused graph (fused GQA, static KV, gate/up without Split) into the volume under /weights/ship/decoder."""
    import onnx
    import ira_onnx as I
    shipped = I.load_shipped(f"{WEIGHTS}/ira/decoder/model.onnx")
    W = I.extract_weights(shipped)
    rc = I.rope_config(f"{WEIGHTS}/rumik_oss_1_config.json")
    out = f"{WEIGHTS}/ship/decoder"
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    I.build_gqa_2bit(shipped, W, rc, f"{out}/model.onnx")
    onnx.checker.check_model(f"{out}/model.onnx", full_check=False)
    VOL.commit()
    sizes = {f: os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)}
    print("[ship] built", sizes, flush=True)
    return sizes


@app.local_entrypoint()
def ship():
    print(build_ship.remote())
