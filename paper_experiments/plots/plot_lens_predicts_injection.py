"""fig_lens_predicts_injection: causal injection profile E_l vs each lens's
predicted deficiency D_l at tau=2 on 2WMH (8B/70B stack). Stars mark each
lens's selected layer; the large star is the causal optimum. Numeric summary
lives in tab:lenspred. Data: tweak_factor_analysis results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "style"))
import lens_style as S

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / "experiments/04_memory_injection/data"
# 8B picks come from the parity250 checkpoints (matches the paper's tables)
PANELS = [
    ("LLaMA-3-8B", DATA_ROOT / "results_llama8b_expanded",
     [("is_parity250", S.LENS_LABELS["is"], S.LENS_COLORS["is"]),
      ("topk_parity250", S.LENS_LABELS["topk"], S.LENS_COLORS["topk"]),
      ("logit", S.LENS_LABELS["logit"], S.LENS_COLORS["logit"])]),
    ("LLaMA-3-70B", DATA_ROOT / "results_llama70b_expanded",
     [("is_r64", S.LENS_LABELS["is"], S.LENS_COLORS["is"]),
      ("logit", S.LENS_LABELS["logit"], S.LENS_COLORS["logit"])]),
]
TAU = "causal_P_ans_tau2.0"


def lens_delta(d, nL):
    t0 = pd.read_csv(d / "tweak_0.csv"); t1 = pd.read_csv(d / "tweak_1.csv")
    D = np.array([t1[f"ans_prob_lens_edit_layer{l}"].mean()
                  - t0[f"ans_prob_lens_edit_layer{l}"].mean() for l in range(nL)])
    return D, t0.answer_prob_obs.mean()


def main():
    fig, axes = plt.subplots(2, 1, figsize=(S.WIDTH_1COL, 5.0), squeeze=False)
    for j, (name, root, lenses) in enumerate(PANELS):
        ax = axes[j][0]
        cp = pd.read_csv(root / "is_r64/2wmh/causal_profile.csv")
        E = cp[TAU].values
        nL = len(E)
        L = np.arange(nL)
        aE = int(E.argmax())

        picks = []
        pobs = None
        for tag, label, color in lenses:
            d = root / tag / "2wmh"
            if not (d / "tweak_1.csv").exists():
                continue
            D, pobs = lens_delta(d, nL)
            aD = int(D.argmax())
            Dn = (D - D.min()) / (D.max() - D.min() + 1e-12)
            # D_l rescaled onto the E_l range; the caption explains this
            ax.plot(L, pobs + Dn * (E.max() - pobs), "--", color=color, lw=1.4,
                    alpha=0.9, label=f"$\\Delta_\\ell$ {label}")
            picks.append((tag, label, color, aD))

        ax.plot(L, E, "-", color=S.DARK, lw=1.8, label="causal $E_\\ell$")
        ax.axhline(pobs, color="k", lw=.8, ls=":", label="$P_{obs}$ (no injection)")
        ax.axvline(nL - 1, color=S.GREY, ls=":", lw=1.2, label="last layer")
        ax.scatter([aE], [E[aE]], marker="*", s=140, color=S.DARK,
                   edgecolors="white", zorder=6)
        for tag, label, color, aD in picks:
            ax.scatter([aD], [E[aD]], marker="*", s=90, color=color,
                       edgecolors="white", zorder=6)
        ax.set_ylabel("$P(\\mathrm{answer})$")
        if j == len(PANELS) - 1:
            ax.set_xlabel("injection layer $\\ell$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout()
    # anchored at the figure bottom so it cannot collide with the xlabel
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, fontsize=S.FONT_ANNOT)
    S.finalize(fig, S.FIGDIR / "fig_lens_predicts_injection", png=True, sync=True)
    print("wrote fig_lens_predicts_injection")


if __name__ == "__main__":
    main()
