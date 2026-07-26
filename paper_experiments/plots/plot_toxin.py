"""fig_toxin_zero / fig_toxin_soft: flagged-head zero-ablation (8B vs 70B) and
the soft-subtraction toxicity/perplexity tradeoff (8B). 8B rows come from the
parity250 Top-k+IS lens's heads (matches tab:ablation)."""
import numpy as np
import pandas as pd

from case_common import TOX, S, label
import matplotlib.pyplot as plt


def deltas(path):
    t = pd.read_csv(path).set_index("config")
    base = t.loc["baseline", "mean_tox"]
    pbase = t.loc["baseline", "ppl"]
    d = {c: 100 * (t.loc[c, "mean_tox"] - base) / base for c in t.index if c != "baseline"}
    p = {c: 100 * (t.loc[c, "ppl"] - pbase) / pbase for c in t.index if c != "baseline"}
    return d, p


def main():
    d8, p8 = deltas(TOX / "results_llama8b/toxin_is_parity250/ablation_summary.csv")
    d70, _ = deltas(TOX / "results_llama70b/toxin/ablation_summary.csv")
    w = 0.34

    fig, ax = plt.subplots(figsize=S.SIZE_1COL, constrained_layout=True)
    x = np.arange(2)
    b1 = ax.bar(x - w / 2, [d8["zero_top15"], d70["zero_top15"]], w,
                label="DART top-15 heads", color=S.ORANGE)
    b2 = ax.bar(x + w / 2, [d8["zero_random15"], d70["zero_random15"]], w,
                label="15 random heads", color=S.GREY)
    label(ax, list(b1) + list(b2), fmt="{:+.1f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, ["LLaMA-3-8B", "LLaMA-3-70B"])
    ax.set_ylabel("$\\Delta$ mean toxicity (%)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.15 * (hi - lo))
    ax.legend(loc="lower right", fontsize=S.FONT_ANNOT)
    S.finalize(fig, S.FIGDIR / "fig_toxin_zero.pdf", sync=True)
    print("wrote fig_toxin_zero")

    fig, ax = plt.subplots(figsize=S.SIZE_1COL, constrained_layout=True)
    lams = sorted(c for c in d8 if c.startswith("soft_lambda"))
    x = np.arange(len(lams))
    b1 = ax.bar(x - w / 2, [d8[c] for c in lams], w, label="$\\Delta$ toxicity", color=S.BLUE)
    b2 = ax.bar(x + w / 2, [p8[c] for c in lams], w, label="$\\Delta$ perplexity", color=S.TUNED_RED)
    label(ax, list(b1) + list(b2), fmt="{:+.1f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"$\\lambda$={c.split('lambda')[1]}" for c in lams])
    ax.set_ylabel("change vs. baseline (%)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.15 * (hi - lo))
    ax.legend(loc="upper left", fontsize=S.FONT_ANNOT)
    S.finalize(fig, S.FIGDIR / "fig_toxin_soft.pdf", sync=True)
    print("wrote fig_toxin_soft")


if __name__ == "__main__":
    main()
