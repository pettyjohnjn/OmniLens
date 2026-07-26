"""fig_trajectory_grid: per-position top-1 trajectory grid (full-rank vs
Top-k+IS, same prompt), after Belrose et al. Data: trajectory_apps grids."""
import json

import numpy as np

from case_common import TRAJ, S
import matplotlib.pyplot as plt


def main():
    g = json.loads((TRAJ / "grids/gpt2_ioi.json").read_text())
    toks = [t.replace(" ", "␣") for t in g["tokens"]]
    panels = [("full-rank tuned", "tuned"), ("OmniLens Top-$k$+IS", "lora_is")]

    fig, axes = plt.subplots(1, 2, figsize=S.SIZE_2COL_TALL, constrained_layout=True)
    for ax, (lbl, lens) in zip(axes, panels):
        rows = g["lenses"][lens]
        P, T = len(rows), len(toks)
        probs = np.array([r["probs"] for r in rows])
        im = ax.pcolormesh(probs, cmap="Blues", vmin=0, vmax=1)
        for i, r in enumerate(rows):
            for j in range(T):
                tok = r["tokens"][j].strip()[:7]
                ax.text(j + 0.5, i + 0.5, tok, ha="center", va="center",
                        fontsize=4.6,
                        color="white" if probs[i, j] > 0.6 else "#263238")
        ax.set_xticks(np.arange(T) + 0.5)
        ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(np.arange(P) + 0.5)
        ax.set_yticklabels(
            ["embed"] + [f"L{r['hs']-1}" for r in rows[1:-1]] + ["output"],
            fontsize=6)
        ax.set_xlabel(lbl)
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="top-1 probability")
    S.finalize(fig, S.FIGDIR / "fig_trajectory_grid.pdf", sync=True)
    print("wrote fig_trajectory_grid")


if __name__ == "__main__":
    main()
