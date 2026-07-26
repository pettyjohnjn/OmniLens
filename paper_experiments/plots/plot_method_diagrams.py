#!/usr/bin/env python3
"""Method schematics:

  fig_method_estimator   Top-k+IS head/tail decomposition
  fig_lens_template      generic lens template for Section 2
  fig_memory_optimizer   optimizer-state bucket, exact parameter math
  fig_memory_readout     measured single-GPU peak vs sequence length

The schematics are hand-laid at a 7pt base with bespoke per-text sizes; the
memory figures read benchmarks/readout_scaling_8b_lse/summary.csv
(re-measured after the teacher-prep patch and the streamed fp32
log-partition; omnilens commits 16314a1 + 7d0b435)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "style"))
import lens_style as S

OUT = S.FIGDIR

ORANGE = "#e6862c"
BLUE = "#2f6f9f"
GREEN = "#3b8f5a"
DARK = "#333333"


def setup():
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7})
    return plt


def fig_estimator():
    plt = setup()

    rng = np.random.default_rng(3)
    V = 64
    p = 1.0 / (np.arange(V) + 2.5) ** 1.35
    p /= p.sum()
    k = 12
    tail_idx = np.sort(rng.choice(np.arange(k, V), size=6, replace=False,
                                  p=p[k:] / p[k:].sum()))

    fig, ax = plt.subplots(figsize=(3.4, 2.15), constrained_layout=True)
    xs = np.arange(V)
    colors = [BLUE if i < k else "#d9d9d9" for i in xs]
    for i in tail_idx:
        colors[i] = GREEN
    ax.bar(xs, p, width=0.85, color=colors, linewidth=0)
    for i in tail_idx:
        ax.plot([i], [p[i] * 1.35], marker="v", color=GREEN, markersize=3.4)

    ax.set_yscale("log")
    ax.set_ylim(p.min() * 0.55, p.max() * 4.5)
    ax.set_xlim(-1.2, V)
    ax.set_xticks([])
    ax.set_xlabel("vocabulary, sorted by teacher probability", fontsize=7)
    ax.set_ylabel(r"$P(t\,|\,x)$", fontsize=7.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.axvspan(-1.2, k - 0.5, color=BLUE, alpha=0.06, zorder=0)
    y_top = p.max() * 3.4
    ax.text(k / 2 - 0.5, y_top,
            "exact head\n" r"$\sum_{i \leq k_{\mathrm{head}}} P \log \frac{P}{Q}$",
            ha="center", va="top", fontsize=6.6, color=BLUE)
    ax.text(V - 1, y_top,
            "sampled tail: $k_{\\mathrm{tail}}$ draws $t_i \\sim R$,\n"
            r"weight $P(t_i)\,/\,R(t_i)$",
            ha="right", va="top", fontsize=6.6, color=GREEN)
    ax.text(V - 1, p.max() * 1.28,
            "teacher-tail default: all weights $= 1 - P_{\\mathrm{head}}$",
            ha="right", va="top", fontsize=6.0, color="#555555")
    ax.annotate("",
                xy=(tail_idx[3] + 0.6, p[tail_idx[3]] * 1.52),
                xytext=(V * 0.62, p.max() * 0.92),
                arrowprops={"arrowstyle": "-|>", "linewidth": 0.8,
                            "color": GREEN, "alpha": 0.65})

    S.finalize(fig, OUT / "fig_method_estimator.pdf", sync=True)
    print("wrote fig_method_estimator")


def fig_lens_template():
    plt = setup()
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    ymain = 2.15

    def box(x0, x1, top, sub, face, edge, accent=False):
        b = FancyBboxPatch((x0, ymain - 0.72), x1 - x0, 1.44,
                           boxstyle="round,pad=0.03",
                           facecolor=face, edgecolor=edge, linewidth=1.1)
        ax.add_patch(b)
        xc = (x0 + x1) / 2
        ax.text(xc, ymain + 0.33, top, ha="center", va="center", fontsize=8.0,
                color=edge if accent else DARK)
        ax.text(xc, ymain - 0.33, sub, ha="center", va="center",
                fontsize=5.6, color="#666666")

    def arrow(x0, x1, y=ymain):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops={"arrowstyle": "-|>", "linewidth": 1.2,
                                "color": DARK})

    ax.text(0.62, ymain + 0.10, r"$h_{\ell,u}$", ha="center", va="center",
            fontsize=8.5)
    ax.text(0.62, ymain - 0.40, "hidden\nstate", ha="center", va="top",
            fontsize=5.6, color="#666666")
    arrow(1.10, 1.62)
    box(1.62, 3.62, r"$\mathcal{L}_{\ell,u}$", "translator\n(trainable)",
        "#fdf3e7", ORANGE, accent=True)
    arrow(3.62, 4.14)
    box(4.14, 5.52, r"$\eta$", "final norm\n(frozen)", "#eef2f7", DARK)
    arrow(5.52, 6.04)
    box(6.04, 7.46, r"$W_U$", "unembed\n(frozen)", "#eef2f7", DARK)
    arrow(7.46, 7.98)
    ax.text(8.92, ymain + 0.10, r"$Q_{\ell,u}(\cdot\,|\,x)$", ha="center",
            va="center", fontsize=8.0)
    ax.text(8.92, ymain - 0.40, "softmax", ha="center", va="top",
            fontsize=5.6, color="#666666")

    ax.text(5.0, 0.42,
            r"trained to match the model's final output:"
            r"  $D_{\mathrm{KL}}\!\left(P(\cdot|x) \,\|\, Q_{\ell,u}\right)$",
            ha="center", va="center", fontsize=6.6, color="#555555")

    S.finalize(fig, OUT / "fig_lens_template.pdf", sync=True)
    print("wrote fig_lens_template")


def fig_memory_split():
    import csv

    plt = setup()

    # optimizer bucket: exact parameter math, AdamW 12 B/param
    models = ["GPT-2\n(74 sites)", "LLaMA-3-8B\n(194 sites)", "LLaMA-3-70B\n(482 sites)"]
    full_opt = [0.5, 39.1, 388.0]
    lora_opt = [0.09, 1.23, 6.1]

    figA, axA = plt.subplots(figsize=(3.4, 2.25), constrained_layout=True)
    x = np.arange(3)
    w = 0.36
    axA.bar(x - w / 2, full_opt, w, color="#8a8a8a", label="full-rank")
    axA.bar(x + w / 2, lora_opt, w, color=ORANGE, label="LoRA $r$64")
    axA.axhline(40, color="#b03a2e", linestyle="--", linewidth=1.1)
    axA.text(-0.55, 50, "A100-40GB", ha="left", fontsize=6.6, color="#b03a2e")
    axA.set_yscale("log")
    axA.set_ylim(0.05, 900)
    axA.set_xticks(x)
    axA.set_xticklabels(models, fontsize=7)
    axA.set_ylabel("memory (GB, log)", fontsize=8)
    axA.legend(frameon=False, fontsize=7, loc="upper left")
    for xi, v in zip(x - w / 2, full_opt):
        axA.text(xi, v * 1.35, f"{v:g}", ha="center", fontsize=6.2, color="#555555")
    axA.grid(True, axis="y", color="#e5e5e5", linewidth=0.7)
    for spine in ("top", "right"):
        axA.spines[spine].set_visible(False)
    S.finalize(figA, OUT / "fig_memory_optimizer.pdf", sync=True)

    # measured readout curves
    bench = (Path(__file__).resolve().parents[1]
             / "experiments/02_memory_systems/data/benchmarks/readout_scaling_8b_lse/summary.csv")
    curves = {"fullkl": [], "topk": [], "is": []}
    ooms = {"fullkl": [], "topk": [], "is": []}
    for row in csv.DictReader(bench.open()):
        s = int(row["seq_len"])
        if row["status"] == "OK":
            curves[row["loss"]].append((s, float(row["peak_gb"])))
        else:
            ooms[row["loss"]].append(s)

    figB, axB = plt.subplots(figsize=(3.4, 2.25), constrained_layout=True)
    styles = {
        "fullkl": ("Full-KL", "#111111", "-"),
        "topk": ("Top-$k$", "#2f6f9f", "--"),
        "is": ("Top-$k$+IS", "#3b8f5a", "-"),
    }
    seqs = [1024, 2048, 4096, 8192]
    dodge = {"fullkl": 0.90, "topk": 1.10, "is": 1.0}
    for tag, (label, color, ls) in styles.items():
        pts = sorted(curves[tag])
        axB.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                 markersize=3.6, color=color, linestyle=ls, linewidth=1.6,
                 label=label)
        if ooms[tag]:
            s0 = min(ooms[tag])
            axB.plot([s0 * dodge[tag]], [41.5], marker="x", markersize=6,
                     color=color, markeredgewidth=1.8, clip_on=False)
    axB.axhline(40, color="#b03a2e", linestyle="--", linewidth=1.1)
    axB.text(1050, 40.8, "A100-40GB  ($\\times$ = OOM)", ha="left",
             fontsize=6.6, color="#b03a2e")
    axB.set_xscale("log", base=2)
    axB.set_xticks(seqs)
    axB.set_xticklabels([str(s) for s in seqs], fontsize=7.5)
    axB.minorticks_off()
    axB.set_ylim(20, 44)
    axB.set_xlabel("sequence length (fixed microbatch of 2)", fontsize=7.5)
    axB.set_ylabel("peak memory (GB)", fontsize=8)
    axB.legend(frameon=False, fontsize=7, loc="lower right")
    axB.grid(True, axis="y", color="#e5e5e5", linewidth=0.7)
    for spine in ("top", "right"):
        axB.spines[spine].set_visible(False)
    S.finalize(figB, OUT / "fig_memory_readout.pdf", sync=True)
    print("wrote fig_memory_optimizer + fig_memory_readout")


if __name__ == "__main__":
    fig_estimator()
    fig_lens_template()
    fig_memory_split()
