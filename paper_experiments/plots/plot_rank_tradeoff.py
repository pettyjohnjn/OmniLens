#!/usr/bin/env python3
"""fig_rank_tradeoff (+2panel variant): fidelity vs rank for Section 3.
Values are the canonical Table-1 numbers (GPT-2 Small, 131,072 Pile test
tokens, step 1000; KL to teacher, agreement vs the full-rank baseline),
from evaluation/gpt2_preemptable_sweep_1000 and the agreement pipelines."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "style"))
import lens_style as S

import matplotlib.pyplot as plt

RANKS = [1, 4, 8, 16, 32, 64, 128, 256, 384]
FINAL_KL = [0.0760, 0.0574, 0.0546, 0.0526, 0.0479, 0.0431, 0.0402, 0.0382, 0.0362]
MEAN_KL = [1.037, 0.878, 0.799, 0.755, 0.696, 0.600, 0.574, 0.555, 0.539]
FINAL_TOP1 = [0.837, 0.853, 0.859, 0.868, 0.878, 0.888, 0.900, 0.907, 0.915]
MEAN_TOP1 = [0.472, 0.535, 0.570, 0.599, 0.655, 0.703, 0.746, 0.773, 0.793]
FINAL_RHO = [0.971, 0.975, 0.978, 0.978, 0.981, 0.984, 0.986, 0.987, 0.989]
MEAN_RHO = [0.787, 0.855, 0.885, 0.903, 0.925, 0.941, 0.952, 0.959, 0.964]
FINAL_TAU = [0.596, 0.638, 0.639, 0.662, 0.670, 0.693, 0.718, 0.739, 0.762]
MEAN_TAU = [0.133, 0.220, 0.262, 0.295, 0.363, 0.423, 0.479, 0.531, 0.574]
BASE_FINAL_KL, BASE_MEAN_KL = 0.0411, 0.543
R_STAR = 64

PANELS = [
    ("KL to teacher", FINAL_KL, MEAN_KL, (BASE_FINAL_KL, BASE_MEAN_KL), "log"),
    ("top-1 agreement", FINAL_TOP1, MEAN_TOP1, None, "linear"),
    (r"Pearson $\rho$", FINAL_RHO, MEAN_RHO, None, "linear"),
    (r"Kendall $\tau$@100", FINAL_TAU, MEAN_TAU, None, "linear"),
]

I_STAR = RANKS.index(R_STAR)


def draw_panel(ax, ylabel, final, mean, base, yscale, xticklabels, ms=3.2):
    # hollow markers throughout; only the recommended r=64 point is filled
    for vals, color, ls, marker, label in (
        (final, S.BLUE, "-", "o", "final layer"),
        (mean, S.ORANGE, "--", "s", "mean (12 layers)"),
    ):
        ax.plot(RANKS, vals, color=color, lw=1.2, ls=ls, marker=marker,
                ms=ms, markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=0.9, label=label)
        ax.plot([R_STAR], [vals[I_STAR]], marker=marker, ms=ms + 0.4,
                color=color, ls="none", zorder=5)
    if base is not None:
        ax.axhline(base[0], color=S.BLUE, lw=0.7, ls=(0, (1, 2)))
        ax.axhline(base[1], color=S.ORANGE, lw=0.7, ls=(0, (1, 2)))
    ax.set_xscale("log", base=2)
    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel, labelpad=1.5)
    ax.set_xticks(RANKS)
    ax.set_xticklabels(xticklabels)
    ax.minorticks_off()
    ax.tick_params(labelsize=7, pad=1.5)
    S.despine(ax)


def main():
    # 2x2 with all four metrics
    sparse = [str(r) if r in (1, 8, 64, 384) else "" for r in RANKS]
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.1), sharex=True)
    for ax, (ylabel, final, mean, base, yscale) in zip(axes.flat, PANELS):
        draw_panel(ax, ylabel, final, mean, base, yscale, sparse, ms=2.8)
    for ax in axes[1]:
        ax.set_xlabel("rank $r$", labelpad=1.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=7, bbox_to_anchor=(0.55, 1.02), handlelength=1.6,
               columnspacing=1.0)
    fig.tight_layout(h_pad=0.5, w_pad=0.8, rect=(0, 0, 1, 0.95))
    S.finalize(fig, S.FIGDIR / "fig_rank_tradeoff", sync=True)

    # 2 stacked full-width panels (KL + top-1)
    dense = [("" if r == 256 else str(r)) for r in RANKS]
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.4), sharex=True)
    for ax, (ylabel, final, mean, base, yscale) in zip(axes, (PANELS[0], PANELS[1])):
        draw_panel(ax, ylabel, final, mean, base, yscale, dense)
    axes[-1].set_xlabel("rank $r$ (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=7, bbox_to_anchor=(0.55, 1.01))
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.5, rect=(0, 0, 1, 0.955))
    S.finalize(fig, S.FIGDIR / "fig_rank_tradeoff_2panel", sync=True)
    print("wrote fig_rank_tradeoff + fig_rank_tradeoff_2panel")


if __name__ == "__main__":
    main()
