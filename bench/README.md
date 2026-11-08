# Decoder attention benchmark (Modal GPU)

Profiles and benchmarks attention variants of the 2-bit Ira decoder
(`Sanyam0605/ira-tts-2bit-web`, `decoder/model.onnx`, 36 layers, GQA 16/4, head dim 128).

Run it (needs `modal` logged in):

```bash
modal run bench/modal_app.py::validate   # dequantisation round-trip + graph sanity, one GPU container
modal run bench/modal_app.py             # all variants in parallel + ORT profiles -> bench/results/
```

## What is compared

| variant | graph | KV cache | ORT attention kernel |
|---|---|---|---|
| `shipped_dense` | the graph as shipped on HF (5,476 nodes, traced MatMul/Softmax attention) | `Concat` every step | explicit ops |
| `gqa_concat` | fused `GroupQueryAttention` per layer (`build_fast_graph.py` design) | `Concat` every step | ORT default (flash / mem-efficient) |
| `gqa_static` | same fused graph | pre-allocated `[1,4,4096,128]` buffer shared by past and present | ORT default |
| `gqa_static_no_flash` | as `gqa_static` | as above | `ORT_DISABLE_FLASH_ATTENTION=1` |
| `gqa_static_unfused` | as `gqa_static` | as above | flash and mem-efficient disabled |

`shipped_dense` is the shipped graph with its 251 `MatMulNBits` nodes replaced by a dense fp16 `MatMul` of the
same dequantised weights. ORT's CUDA EP has no 2-bit kernel (`MatMulNBits` only supports 4 and 8 bits), so the
2-bit weights cannot run on CUDA as shipped. The attention variants do not depend on the weight format, so the
comparison is still valid, but absolute times are for dense fp16 weights, not for 2-bit weights.

Dequantisation is checked against ORT's own `MatMulNBits` kernel (4-bit, same unpacking code path): relative
error 2.7e-4, which is fp16 rounding.

Decode is one token per step with IOBinding and a device-resident KV cache, which is what the app does after
prefill. Token ids are fixed random ids, so the same inputs go to every variant. Correctness is checked by comparing
each variant's logits for the first 64 steps against `shipped_dense`.

## Caveats

* This is CUDA (A10G, sm86), not WebGPU. The browser app runs ORT-web on WebGPU, where the fused GQA kernel is a
  different implementation. Attention-kernel conclusions transfer only as directions, not as numbers.
* ORT's per-node profile for CUDA EP is host-observed kernel time. Treat its split as relative. The per-step
  wall-clock numbers are the reliable ones.
* Decode lengths go to 2,080 tokens. Sliding-window layers use a 4,096 window, which never truncates at these lengths,
  so `local_window_size=-1` is correct here.

## Results (A10G, CUDA EP, onnxruntime-gpu 1.23.2, dense fp16 weights)

Mean ms per decode step over a 32-step window at each context length. Speed-up is relative to `shipped_dense` at 2,048 tokens.

| variant | L=64 | L=512 | L=1024 | L=2048 | speed-up @2048 | top-1 vs shipped (64 steps) | max abs logit diff |
|---|---|---|---|---|---|---|---|
| shipped_dense | 17.08 | 18.41 | 19.24 | 21.08 | 1.00x | 1.000 | 0 |
| gqa_concat | 14.79 | 15.52 | 16.04 | 16.73 | 1.26x | 0.938 | 0.028 |
| gqa_static | 14.55 | 15.84 | 15.61 | 16.14 | 1.31x | 0.938 | 0.028 |
| gqa_static_no_flash | 17.89 | 18.50 | 19.01 | 20.19 | 1.04x | 0.938 | 0.028 |
| gqa_static_unfused | not run: ORT's CUDA GroupQueryAttention has no unfused path | | | | | | |

Notes:
* Flash attention is what makes the fused op fast here: disabling it removes almost the whole gain.
* Static vs concat KV cache makes little difference at these lengths (within noise).
* Top-1 agreement of 0.938 (4 of 64 steps differ vs the shipped graph) is fp16 noise. Against an fp32 PyTorch reference (`bench/results/accuracy_fp32.json`, 64 steps, random token ids):
  shipped fp16 top-1 0.969 (max logit err 0.026), fused nosplit fp16 top-1 0.938 (max logit err 0.019). Every mismatch picks a token within 0.007 logits of the fp32 best. Small sample.
* Round 2: `gqa_static_nosplit` 15.69 ms at 2048 tokens (1.34x). The CUDA-graph variants failed (ORT could not convert an output to numpy) and are unmeasured.
* Attention is only 13–16% of profiled time. Most of the step is the MatMul/Split/Cast work of the weight projections, so attention alone cannot give more than about 1.3x here.
* Profile per-op times are host-observed for CUDA and read only as a relative split. Raw data: `results/bench.json`, `results/profile.json`, `results/run_log.txt`.
