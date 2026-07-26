#!/usr/bin/env python3
"""lens_param_explosion (paper Figure 1): dense-lens-set parameter cost vs
hidden dimension, full-rank (dashed) against LoRA r=64 (solid). Dense set =
(6L+2) translators. Self-contained (model list + formulas). The paper embeds
the .png, so both png and pdf are synced."""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "style"))
import lens_style as S

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_OUT = S.FIGDIR / "lens_param_explosion.png"
RANK = 64

# name, hidden dim d, layers L, family, annotate
MODELS = [
    ("BERT-base", 768, 12, "bertgpt", False),
    ("BERT-large", 1024, 24, "bertgpt", False),
    ("GPT-2 Small", 768, 12, "bertgpt", True),
    ("GPT-2 Medium", 1024, 24, "bertgpt", False),
    ("GPT-2 Large", 1280, 36, "bertgpt", False),
    ("GPT-2 XL", 1600, 48, "bertgpt", True),
    ("GPT-3", 12288, 96, "bertgpt", True),
    ("Pythia-1.4B", 2048, 24, "other", False),
    ("Pythia-2.8B", 2560, 32, "other", False),
    ("Gemma-7B", 3072, 28, "other", False),
    ("Mistral-7B", 4096, 32, "other", False),
    ("Falcon-7B", 4544, 32, "other", False),
    ("LLaMA-3 8B", 4096, 32, "llama", True),
    ("LLaMA-2 13B", 5120, 40, "llama", False),
    ("LLaMA-3 70B", 8192, 80, "llama", True),
    ("LLaMA-3 405B", 16384, 126, "llama", True),
]

# neutral marker tones; the lens palette stays reserved for the trend lines
FAMILY_STYLE = {
    "bertgpt": ("o", "#5a6b7a", "BERT / GPT Family"),
    "llama": ("^", S.DARK, "LLaMA Family"),
    "other": ("s", "#9aa0a6", "Other Open Models"),
}


def n_lenses(layers):
    return 6 * layers + 2


def full_rank_params(d, layers):
    return n_lenses(layers) * d * d


def lora_params(d, layers, rank=RANK):
    return n_lenses(layers) * 2 * rank * d


def depth_trend(ds):
    """Empirical depth trend L(d) fit over the plotted models."""
    xs = np.log([m[1] for m in MODELS])
    ys = np.log([m[2] for m in MODELS])
    slope, intercept = np.polyfit(xs, ys, 1)
    return np.exp(intercept) * ds ** slope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=S.SIZE_2COL, constrained_layout=True)

    ds = np.linspace(256, 17200, 400)
    ls = depth_trend(ds)
    ax.plot(ds, (6 * ls + 2) * ds * ds, linestyle="--", color=S.TUNED_RED, zorder=1)
    ax.plot(ds, (6 * ls + 2) * 2 * RANK * ds, linestyle="-", color=S.ORANGE, zorder=1)

    annotations = {
        "GPT-2 Small": (-10, 26, "right"),
        "GPT-2 XL": (10, 40, "left"),
        "GPT-3": (-58, 30, "right"),
        "LLaMA-3 8B": (30, 24, "left"),
        "LLaMA-3 70B": (46, -22, "left"),
        "LLaMA-3 405B": (-40, -46, "right"),
    }
    for name, d, layers, family, annotate in MODELS:
        marker, color, _ = FAMILY_STYLE[family]
        full = full_rank_params(d, layers)
        ax.scatter([d], [full], marker=marker, s=64, color=color, alpha=1.0, zorder=3)
        ax.scatter([d], [lora_params(d, layers)], marker=marker, s=64, color=color, alpha=0.45, zorder=2)
        if annotate:
            dx, dy, ha = annotations[name]
            ax.annotate(name, xy=(d, full), xytext=(dx, dy),
                        textcoords="offset points", ha=ha,
                        fontsize=S.FONT_ANNOT, color=S.DARK,
                        arrowprops={"arrowstyle": "-", "linewidth": 0.7,
                                    "color": "#aaaaaa"})

    ax.set_xlabel("Hidden Dimension  $d$")
    ax.set_ylabel("Lens Parameters")
    ax.set_xlim(0, 17200)
    ax.set_ylim(-6e9, 2.2e11)
    ax.set_xticks([0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384])
    ax.set_xticklabels(["0", "2k", "4k", "6k", "8k", "10k", "12k", "14k", "16k"])
    ax.set_yticks([0, 5e10, 1e11, 1.5e11, 2e11])
    ax.set_yticklabels(["0", "50B", "100B", "150B", "200B"])

    handles = [
        Line2D([], [], marker=m, linestyle="None", markersize=8, color=c, label=lbl)
        for m, c, lbl in FAMILY_STYLE.values()
    ]
    handles.extend([
        Line2D([], [], linestyle="--", color=S.TUNED_RED, label="Full-Rank"),
        Line2D([], [], linestyle="-", color=S.ORANGE, label=f"OmniLens ($r$={RANK})"),
    ])
    ax.legend(handles=handles, loc="upper left")

    base = args.out.with_suffix("")
    pdf = S.finalize(fig, base, png=True, sync=True)
    # finalize's sync copies the pdf only; the paper embeds the png
    png = base.with_suffix(".png")
    for tree in S.PAPER_TREES:
        if tree.exists():
            shutil.copy2(png, tree / png.name)
    print(f"wrote {pdf} (pdf+png synced)")


if __name__ == "__main__":
    main()
