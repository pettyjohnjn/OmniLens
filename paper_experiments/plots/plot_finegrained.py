"""fig_finegrained_bars: prediction-depth agreement + CBE transfer strength.
Not embedded in the paper (kept for reference; not synced to the paper trees).
Data: trajectory_apps prediction_depth.csv + cbe.csv."""
import numpy as np
import pandas as pd

from case_common import TRAJ, S, label, tag
import matplotlib.pyplot as plt


def main():
    lens_order = [("logit", "logit lens", S.GREY),
                  ("lora_is", "OmniLens Top-$k$+IS", S.ORANGE),
                  ("lora_topk", "OmniLens Top-$k$", S.BLUE)]
    panels = [("GPT-2", "gpt2"), ("LLaMA-3-8B", "llama8b")]

    fig, axes = plt.subplots(1, 2, figsize=S.SIZE_2COL, constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(panels))
    w = 0.8 / len(lens_order)
    for k, (lens, lbl, color) in enumerate(lens_order):
        vals = []
        for _, t in panels:
            d = pd.read_csv(TRAJ / f"results_{t}/prediction_depth.csv")
            vals.append(d[d.lens == lens].within1_pct.mean())
        bars = ax.bar(x + (k - 1) * w, vals, w, label=lbl, color=color)
        label(ax, bars, fmt="{:.0f}")
    ax.set_xticks(x, [m for m, _ in panels])
    ax.set_ylim(0, 100)
    ax.set_ylabel("within 1 hidden state (%)")
    tag(ax, "(A) prediction depth")

    ax = axes[1]
    cbe_order = lens_order + [("tuned", "full-rank tuned", S.TUNED_RED)]
    w = 0.8 / len(cbe_order)
    for k, (lens, lbl, color) in enumerate(cbe_order):
        vals = []
        for _, t in panels:
            d = pd.read_csv(TRAJ / f"results_cbe_{t}/cbe.csv")
            vals.append(d[(d.lens == lens) & (d.j < 8)].model_kl.mean())
        bars = ax.bar(x + (k - 1.5) * w, vals, w, label=lbl, color=color)
        label(ax, bars, fmt="{:.2f}")
    for g, (_, t) in enumerate(panels):
        d = pd.read_csv(TRAJ / f"results_cbe_{t}/cbe.csv")
        r = d[d.lens == "random"].model_kl.mean()
        ax.hlines(r, x[g] - 0.45, x[g] + 0.45, color="#37474F", ls="--", lw=1.2,
                  label="random directions" if g == 0 else None)
    ax.set_xticks(x, [m for m, _ in panels])
    ax.set_ylabel("mean model KL (nats)")   # top-8 directions
    tag(ax, "(B) CBE transfer")

    handles, labels = ax.get_legend_handles_labels()
    axes[0].legend(handles, labels, ncol=3, loc="upper center",
                   bbox_to_anchor=(1.05, -0.14), fontsize=S.FONT_ANNOT)
    S.finalize(fig, S.FIGDIR / "fig_finegrained_bars.pdf")
    print("wrote fig_finegrained_bars")


if __name__ == "__main__":
    main()
