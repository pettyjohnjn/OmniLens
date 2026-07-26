"""fig_case_study_summary: full-width three-panel summary for the case-study
section. (a) prompt-injection detection AUROC per lens and model;
(b) causal gain captured by each selector's injection-layer pick at tau=4
(full sweep in tab:lenspred); (c) GPT-2 whole-model ablation audit;
(d) 8B detection score vs intervention effect per hookpoint type
(mean per-site toxic count reproduces the paper's Spearman -0.43). One
row of four with a shared legend; the standalone figures remain the
appendix versions."""
import csv
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from case_common import CHOOSERS, TOX, TRAJ, S, grouped, label, load_summary, tag
import matplotlib.pyplot as plt
from plot_allsites_ablation import (cost_constrained, load_gpt2_allhooks,
                                    load_gpt2_perhead)


def panel_detection(ax):
    order = [("logit", "logit lens", S.GREY),
             ("lora_is", "OmniLens Top-$k$+IS", S.ORANGE),
             ("lora_topk", "OmniLens Top-$k$", S.BLUE),
             ("tuned", "full-rank ref.", S.TUNED_RED)]
    groups = [("GPT-2", "results_gpt2"), ("LLaMA-3-8B", "results_llama8b"),
              ("LLaMA-3-70B", "results_llama70b")]
    x = np.arange(len(groups))
    w = 0.8 / len(order)
    seen = set()
    for g, (model, root) in enumerate(groups):
        df = pd.read_csv(TRAJ / root / "detection_auroc.csv")
        df = df[df.detector == "lof"]
        for k, (lens, lbl, color) in enumerate(order):
            v = df[df.lens == lens].auroc.values
            if not len(v):
                continue
            pos = x[g] + (k - 1.5) * w
            ax.bar([pos], [v.mean()], w, color=color,
                   label=None if lens in seen else lbl)
            seen.add(lens)
            ax.scatter(np.full(len(v), pos), v, s=8, facecolors="none",
                       edgecolors="#37474F", linewidths=0.6, zorder=3)
    ax.set_xticks(x, ["GPT-2", "8B", "70B"])
    ax.set_ylim(0.5, 1.04)
    ax.set_ylabel("detection AUROC", fontsize=S.FONT_SMALL)


def panel_selection(ax):
    df = load_summary().assign(tau=lambda d: d.tau.astype(str))
    choosers = CHOOSERS[:2] + \
        [(("lens:tuned_parity250",), "full-rank ref.", S.TUNED_RED, None)] + \
        CHOOSERS[2:]
    groups = [("LLaMA-3-8B", "4.0"), ("LLaMA-3-70B", "4.0")]
    grouped(ax, df, "gain_pct", choosers=choosers, groups=groups,
            group_labels=["8B", "70B"], labels=False)
    ax.set_ylim(0, 105)
    ax.set_ylabel("gain captured (%)", fontsize=S.FONT_SMALL)
    tag(ax, "2WMH, $\\tau{=}4$")


def panel_audit(ax, vals, ph, title):
    # the original attention-head-only method is a reference baseline, not a
    # seventh component type: a labeled dashed line, not a bar
    x = np.arange(len(S.SITE_TYPES))
    bars = ax.bar(x, vals, 0.62, color=S.ORANGE)
    label(ax, bars, fmt="{:+.0f}")
    ax.axhline(ph, color="0.35", linestyle=(0, (4, 2)), linewidth=1.2, zorder=0)
    ax.text(len(S.SITE_TYPES) / 2 - 0.5, ph - 1.2, f"attn heads: {ph:+.0f}%",
            ha="center", va="top", color="0.30", fontsize=S.FONT_ANNOT)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, S.SITE_TYPES, fontsize=S.FONT_ANNOT, rotation=40, ha="right")
    ax.set_ylabel("tox. reduction (%)", fontsize=S.FONT_SMALL)
    ax.text(0.5, 1.04, title, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=S.FONT_ANNOT)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.22 * (hi - lo))


LABEL_OFF = {"attn_in": (-0.005, -4.2, "right"), "attn_out": (0.005, 1.2, "left"),
             "resid_mid": (0.005, 1.2, "left"), "mlp_in": (0.006, -2.2, "left"),
             "mlp_out": (0.005, -6.0, "left"), "resid_post": (0.005, 1.2, "left")}


def panel_scatter(ax):
    scores = defaultdict(list)
    for r in csv.DictReader(open(TOX / "results_llama8b/allhooks/toxic_sites_is.csv")):
        scores[r["site_id"].split(".", 1)[1]].append(float(r["score"]))
    from plot_allsites_ablation import cost_constrained, load_8b_allhooks
    det = {st: sum(v) / len(v) for st, v in scores.items()}
    red = {st: cost_constrained(*load_8b_allhooks(st)) for st in S.SITE_TYPES}
    # highlight only the inversion that matters; neutral grey elsewhere
    focus = {"mlp_out": S.PURPLE, "mlp_in": S.RED}
    for st in S.SITE_TYPES:
        ax.scatter(det[st], red[st], s=34, color=focus.get(st, "0.65"), zorder=3)
        dx, dy, ha = LABEL_OFF[st]
        ax.text(det[st] + dx, red[st] + dy, st, fontsize=S.FONT_ANNOT, ha=ha)
    ax.axhline(0, color="k", lw=0.8)
    rho = stats.spearmanr([det[st] for st in S.SITE_TYPES],
                          [red[st] for st in S.SITE_TYPES])[0]
    ax.text(0.02, 0.97, f"Spearman ${rho:+.2f}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=S.FONT_ANNOT)
    ax.set_xlabel("detected toxic signal", fontsize=S.FONT_SMALL)
    ax.text(0.5, 1.04, "8B: detection vs. intervention", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=S.FONT_ANNOT)
    ax.set_ylabel("tox. reduction (%)", fontsize=S.FONT_SMALL)
    ax.set_xlim(0.0, 0.28)
    ax.set_ylim(-11, 48)


def main():
    fig = plt.figure(figsize=(S.WIDTH_2COL, 2.1), constrained_layout=True)
    axes = fig.subplots(1, 4)
    ax_a, ax_b, ax_c, ax_d = axes

    panel_detection(ax_a)
    panel_selection(ax_b)

    gpt2_vals = [cost_constrained(*load_gpt2_allhooks(st)) for st in S.SITE_TYPES]
    gpt2_ph = cost_constrained(*load_gpt2_perhead())
    panel_audit(ax_c, gpt2_vals, gpt2_ph, "GPT-2: intervention by type")
    panel_scatter(ax_d)

    for ax, letter in zip(axes, ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(-0.32, 1.04, letter, transform=ax.transAxes,
                ha="left", va="bottom")

    from matplotlib.patches import Patch
    handles = [Patch(color=S.ORANGE, label="OmniLens Top-$k$+IS"),
               Patch(color=S.BLUE, label="OmniLens Top-$k$"),
               Patch(color=S.TUNED_RED, label="full-rank ref."),
               Patch(color=S.GREY, label="logit lens"),
               Patch(facecolor=S.HEUR_FACE, hatch="//", edgecolor="white",
                     label="final-layer selector")]
    fig.legend(handles=handles, ncol=5, loc="outside lower center",
               fontsize=S.FONT_ANNOT, frameon=False)

    S.finalize(fig, S.FIGDIR / "fig_case_study_summary.pdf", sync=True)
    print("wrote fig_case_study_summary")


if __name__ == "__main__":
    main()
