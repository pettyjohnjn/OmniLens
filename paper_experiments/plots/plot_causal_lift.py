"""fig_causal_lift_bars: best-layer causal lift of memory injection per
model/dataset. Data: tweak_factor_analysis causal profiles."""
import numpy as np
import pandas as pd

from case_common import INJ, MODELS, S
import matplotlib.pyplot as plt


def main():
    lifts, nulls = {}, {}
    for m, root in MODELS.items():
        for ds in ("hand", "2wmh"):
            d = INJ / root / "is_r64" / ds
            cp = pd.read_csv(d / "causal_profile.csv")
            pobs = pd.read_csv(d / "tweak_0.csv").answer_prob_obs.mean()
            pexp = pd.read_csv(d / "tweak_1.csv").answer_prob_exp.mean()
            E = cp[[c for c in cp.columns if c.startswith("causal_P_ans")]].values
            lifts[(m, ds)] = E.max() / pobs
            nulls[(m, ds)] = pexp < pobs      # explicit worse than implicit

    fig, ax = plt.subplots(figsize=S.SIZE_1COL, constrained_layout=True)
    x = np.arange(len(MODELS))
    w = 0.36
    for k, (ds, lbl, color) in enumerate(
            [("hand", "control set", S.GREY), ("2wmh", "2WikiMultiHop", S.BLUE)]):
        vals = [lifts[(m, ds)] for m in MODELS]
        bars = ax.bar(x + (k - 0.5) * w, vals, w, label=lbl, color=color)
        for b, m in zip(bars, MODELS):
            txt = f"{b.get_height():.2f}×" + ("\n(null)" if nulls[(m, ds)] else "")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), txt,
                    ha="center", va="bottom", fontsize=S.FONT_ANNOT)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xticks(x, list(MODELS))
    ax.set_ylim(0, max(lifts.values()) * 1.22)
    ax.set_ylabel("causal lift  $\\max_\\ell E_\\ell \\,/\\, P_{obs}$")
    ax.legend(loc="upper left")
    S.finalize(fig, S.FIGDIR / "fig_causal_lift_bars.pdf", sync=True)
    print("wrote fig_causal_lift_bars")


if __name__ == "__main__":
    main()
