"""Plots for bench/results: step latency vs context, speed-up at 2048, and per-op profile split."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent / "results"
bench = json.loads((R / "bench.json").read_text())
prof = json.loads((R / "profile.json").read_text())
Ls = [64, 512, 1024, 2048]

# 1) latency vs context length
fig, ax = plt.subplots(figsize=(7, 4.5))
for r in bench:
    ys = [r["lat_mean_ms"][str(L)] for L in Ls]
    ax.plot(Ls, ys, marker="o", label=r["name"])
ax.set_xlabel("context length (tokens)"); ax.set_ylabel("ms per decode step (mean of 32 steps)")
ax.set_title("Decode step latency, A10G, CUDA EP (dense fp16 weights)")
ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(R / "latency_vs_context.png", dpi=130); plt.close(fig)

# 2) speed-up at 2048 vs shipped
base = next(r for r in bench if r["name"] == "shipped_dense")["lat_mean_ms"]["2048"]
names = [r["name"] for r in bench]
sp = [base / r["lat_mean_ms"]["2048"] for r in bench]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(names, sp, color=["#888" if n == "shipped_dense" else "#2a7" if n == "gqa_static_nosplit" else "#58a" for n in names])
for b, v in zip(bars, sp):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}x", ha="center", fontsize=9)
ax.axhline(1, color="k", lw=0.8, ls="--")
ax.set_ylabel("speed-up vs shipped_dense @ 2048 tokens"); ax.set_title("Speed-up at 2048 tokens")
plt.xticks(rotation=20, ha="right", fontsize=8); fig.tight_layout(); fig.savefig(R / "speedup_2048.png", dpi=130); plt.close(fig)

# 3) per-op profile split (host-observed, relative)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, p in zip(axes, prof):
    ops = p["top_ops"][:10][::-1]
    ax.barh([o[0] for o in ops], [o[2] * 100 for o in ops], color="#58a")
    ax.set_title(f"{p['name']}: top ops, {p['steps']} steps\n(attention-named share {p['attn_share_by_name']*100:.0f}%)", fontsize=9)
    ax.set_xlabel("% of profiled CUDA time (host-observed)")
    ax.tick_params(axis="y", labelsize=8)
fig.tight_layout(); fig.savefig(R / "profile_top_ops.png", dpi=130); plt.close(fig)
print("wrote plots to", R)
