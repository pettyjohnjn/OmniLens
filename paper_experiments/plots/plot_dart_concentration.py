"""fig_dart_concentration_bars: top-5 head share of DART toxic-head mass per
lens and model. Data: dart_crosslens_summary_parity250.csv."""
import numpy as np
import pandas as pd

from case_common import TOX, S, label
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv(TOX / "dart_crosslens_summary_parity250.csv")
    df = df.assign(lens=df.lens.replace({"is_parity250": "is_r64",
                                         "topk_parity250": "topk_r64",
                                         "tuned_parity250": "tuned"}))
    order = [("logit", "logit lens", S.GREY, None),
             ("is_r64", "OmniLens Top-$k$+IS", S.ORANGE, None),
             ("topk_r64", "OmniLens Top-$k$", S.BLUE, None),
             ("tuned", "full-rank, full KL", S.TUNED_RED, None)]
    models = ["GPT-2", "LLaMA-3-8B", "LLaMA-3-70B"]
    fig, ax = plt.subplots(figsize=S.SIZE_1COL, constrained_layout=True)
    x = np.arange(len(models))
    w = 0.2
    for k, (tag_, lbl, color, hatch) in enumerate(order):
        vals, pos = [], []
        for g, m in enumerate(models):
            r = df[(df.model == m) & (df.lens == tag_)]
            if len(r):
                vals.append(float(r.top5_share.iloc[0]))
                pos.append(x[g] + (k - 1.5) * w)
        bars = ax.bar(pos, vals, w, label=lbl, color=color, hatch=hatch)
        label(ax, bars, fmt="{:.0f}")
    ax.set_xticks(x, models)
    ax.set_ylabel("top-5 head share (%)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=S.FONT_ANNOT)
    S.finalize(fig, S.FIGDIR / "fig_dart_concentration_bars.pdf", sync=True)
    print("wrote fig_dart_concentration_bars")


if __name__ == "__main__":
    main()
