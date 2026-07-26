#!/usr/bin/env python3
"""fig_rank_structure: why low rank is a constraint, not a compression.

(a) KL vs rank on one protocol (GPT-2 Small, step 1000, 131,072 shuffled
    Pile-val tokens): a lens TRAINED at rank r against the same full-rank lens
    TRUNCATED to rank r by SVD, plus a random-r-subspace control.
(b) Frobenius energy of the learned deviation captured at rank r, from the
    spectral study -- the matrix view, which says rank 64 keeps only 59%.

Panel (a) is measured KL and panel (b) is matrix reconstruction; together they
say the deviation is not low rank as a matrix, yet training under a rank-64
constraint still reaches full-rank fidelity.

Data: experiments/06_rank_structure/data/."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "style"))
import lens_style as S

import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parents[1] / "experiments/06_rank_structure/data"
RANKS = [1, 4, 8, 16, 32, 64, 128, 256, 384]
TICKS = [1, 4, 16, 64, 256]
R_STAR = 64


def load_kl():
    by = defaultdict(dict)
    with open(DATA / "gpt2_rank_truncation.csv") as f:
        for row in csv.DictReader(f):
            by[row["variant"]][int(row["layer"])] = float(row["kl"])
    fin = max(by["full_rank"])
    return {k: v[fin] for k, v in by.items()}


def load_energy():
    """Mean rank-r Frobenius energy fraction at the final checkpoint."""
    rows = [r for r in csv.DictReader(open(DATA / "gpt2_spectrum_layer_summary.csv"))]
    last = max(int(r["checkpoint_step"]) for r in rows)
    rows = [r for r in rows if int(r["checkpoint_step"]) == last]
    out = []
    for r in RANKS:
        col = f"variance_at_rank_{r}"
        vals = [float(x[col]) for x in rows if col in x and x[col]]
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def main():
    kl = load_kl()
    energy = load_energy()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(S.WIDTH_2COL, 2.3),
                                   constrained_layout=True)

    series = [("trained_r{}", "trained at rank $r$", S.ORANGE, "-", "o"),
              ("truncated_r{}", "full-rank, truncated to $r$", S.BLUE, "--", "s"),
              ("random_r{}", "random $r$-dim subspace", S.GREY, ":", "^")]
    for tmpl, label, color, ls, mk in series:
        ys = [kl[tmpl.format(r)] for r in RANKS]
        axA.plot(RANKS, ys, color=color, lw=1.2, ls=ls, marker=mk, ms=3.2,
                 markerfacecolor="white", markeredgecolor=color,
                 markeredgewidth=0.9, label=label)
    axA.axhline(kl["full_rank"], color="0.35", lw=1.0, ls=(0, (4, 2)), zorder=0)
    axA.text(1.15, kl["full_rank"] * 0.93, "full-rank lens", fontsize=S.FONT_ANNOT,
             color="0.30", va="top", ha="left")
    axA.set_xscale("log", base=2); axA.set_yscale("log")
    axA.set_ylim(top=9.0)   # headroom so the legend clears the random curve
    axA.set_xticks(TICKS); axA.set_xticklabels([str(r) for r in TICKS],
                                               fontsize=S.FONT_ANNOT)
    axA.minorticks_off()
    axA.set_xlabel("rank $r$", fontsize=S.FONT_SMALL)
    axA.set_ylabel("final-layer KL to teacher", fontsize=S.FONT_SMALL)
    axA.legend(fontsize=S.FONT_ANNOT, frameon=False, loc="upper right",
               labelspacing=0.25, borderpad=0.1)

    axB.plot(RANKS, energy, color=S.BLUE, lw=1.2, marker="s", ms=3.2,
             markerfacecolor="white", markeredgecolor=S.BLUE, markeredgewidth=0.9)
    i = RANKS.index(R_STAR)
    axB.plot([R_STAR], [energy[i]], marker="s", ms=3.8, color=S.BLUE)
    axB.annotate(f"$r{{=}}{R_STAR}$: {energy[i]*100:.0f}%",
                 xy=(R_STAR, energy[i]), xytext=(6, -14),
                 textcoords="offset points", fontsize=S.FONT_ANNOT, color=S.BLUE)
    axB.set_xscale("log", base=2)
    axB.set_xticks(TICKS); axB.set_xticklabels([str(r) for r in TICKS],
                                               fontsize=S.FONT_ANNOT)
    axB.minorticks_off()
    axB.set_ylim(0, 1.02)
    axB.set_xlabel("rank $r$", fontsize=S.FONT_SMALL)
    axB.set_ylabel("Frobenius energy of $\\Delta$ at rank $r$", fontsize=S.FONT_SMALL)

    for ax, letter in [(axA, "(a)"), (axB, "(b)")]:
        ax.text(-0.16, 1.03, letter, transform=ax.transAxes, ha="left", va="bottom")

    S.finalize(fig, S.FIGDIR / "fig_rank_structure.pdf", sync=True)
    print("wrote fig_rank_structure")


if __name__ == "__main__":
    main()
