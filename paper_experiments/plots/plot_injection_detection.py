"""fig_injection_detection_bars: prompt-injection detection AUROC per lens and
model. Bars = mean LOF AUROC over the ten tasks; open circles = individual
tasks (the low ones are the knowledge tasks). Means live in tab:inj_detect.
Data: trajectory_apps detection_auroc.csv."""
import numpy as np
import pandas as pd

from case_common import TRAJ, S
import matplotlib.pyplot as plt


def main():
    order = [("logit", "logit lens", S.GREY),
             ("lora_is", "OmniLens Top-$k$+IS", S.ORANGE),
             ("lora_topk", "OmniLens Top-$k$", S.BLUE),
             ("tuned", "full-rank tuned", S.TUNED_RED)]
    groups = [("GPT-2", "results_gpt2"), ("LLaMA-3-8B", "results_llama8b"),
              ("LLaMA-3-70B", "results_llama70b")]

    fig, ax = plt.subplots(figsize=S.SIZE_1COL, constrained_layout=True)
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
            ax.scatter(np.full(len(v), pos), v, s=10, facecolors="none",
                       edgecolors="#37474F", linewidths=0.7, zorder=3)
    ax.set_xticks(x, [m for m, _ in groups])
    ax.set_ylim(0.5, 1.04)
    ax.set_ylabel("detection AUROC (LOF)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=S.FONT_ANNOT)
    S.finalize(fig, S.FIGDIR / "fig_injection_detection_bars.pdf", sync=True)
    print("wrote fig_injection_detection_bars")


if __name__ == "__main__":
    main()
