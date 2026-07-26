"""LLaMA case-study figures from the result CSVs (no GPU needed):

  fig_injection_heatmaps_8b/70b   P(answer) over (layer x tweak), hand/2wmh stack
  fig_injection_curves            best-layer lift vs tweak, all model/dataset
  fig_dart_heatmap_8b/70b         DART toxic-count [layer x head]
  fig_dart_layerprofile           summed toxic count per layer, 8B/70B stack

The toxin-ablation figure lives in plot_toxin.py, not here.
Data: tweak_factor_analysis + toxicity_ablation result CSVs."""
import argparse
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "style"))
import lens_style as S

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1] / "experiments"
INJ = EXP / "04_memory_injection/data"
DART = EXP / "05_toxicity_dart_toxin/data"

HEATMAP_CMAP = "viridis"


def inj_matrix(model_dir, ds):
    """[n_layers, n_tweaks] mean P(answer) through the lens."""
    files = sorted(glob.glob(f"{model_dir}/is_r64/{ds}/tweak_*.csv"),
                   key=lambda p: int(re.findall(r"tweak_(\d+)", p)[0]))
    if not files:
        return None
    tweaks = [int(re.findall(r"tweak_(\d+)", f)[0]) for f in files]
    d0 = pd.read_csv(files[0])
    nL = sum(c.startswith("ans_prob_lens_edit_layer") for c in d0.columns)
    M = np.zeros((nL, len(files)))
    for j, f in enumerate(files):
        dd = pd.read_csv(f)
        for l in range(nL):
            M[l, j] = dd[f"ans_prob_lens_edit_layer{l}"].mean()
    return M, d0.answer_prob_obs.mean(), d0.answer_prob_exp.mean(), tweaks


def fig_injection_heatmaps(out):
    specs = [("8b", INJ / "results_llama8b_expanded"), ("70b", INJ / "results_llama70b_expanded")]
    dss = ["hand", "2wmh"]
    for tag, mdir in specs:
        fig, axes = plt.subplots(2, 1, figsize=S.SIZE_1COL_STACK, squeeze=False)
        for i, ds in enumerate(dss):
            ax = axes[i][0]
            r = inj_matrix(str(mdir), ds)
            if r is None:
                ax.set_visible(False); continue
            M, pobs, pexp, tweaks = r
            im = ax.imshow(M, aspect="auto", origin="lower", cmap=HEATMAP_CMAP,
                           extent=[tweaks[0]-.5, tweaks[-1]+.5, -.5, M.shape[0]-.5])
            ax.grid(False)
            bl, bt = np.unravel_index(M.argmax(), M.shape)
            ax.scatter([tweaks[bt]], [bl], marker="*", s=90, c=S.RED, edgecolors="white",
                       linewidths=0.6)
            if i == 1:
                ax.set_xlabel("tweak factor $\\tau$")
            ax.set_ylabel("injection layer")
            fig.colorbar(im, ax=ax, fraction=.046, label="P(answer)")
        fig.tight_layout()
        S.finalize(fig, out / f"fig_injection_heatmaps_{tag}", png=True, sync=True)
        print(f"wrote fig_injection_heatmaps_{tag}")


def fig_injection_curves(out):
    fig, ax = plt.subplots(figsize=S.SIZE_1COL)
    model_colors = {"8B": S.BLUE, "70B": S.ORANGE}
    for name, mdir in [("8B", INJ/"results_llama8b_expanded"), ("70B", INJ/"results_llama70b_expanded")]:
        for ds, ls in [("2wmh", "-"), ("hand", "--")]:
            r = inj_matrix(str(mdir), ds)
            if r is None: continue
            M, pobs, pexp, tweaks = r
            best = M.max(axis=0) / max(pobs, 1e-9)
            ax.plot(tweaks, best, ls, marker="o", ms=3, color=model_colors[name],
                    label=f"{name} {ds}")
    ax.axhline(1.0, color=S.GREY, lw=.8, ls=":")
    ax.set_xlabel("tweak factor $\\tau$"); ax.set_ylabel("best-layer lift  $P^*/P_{obs}$")
    ax.legend()
    fig.tight_layout()
    S.finalize(fig, out / "fig_injection_curves", png=True, sync=True)
    print("wrote fig_injection_curves")


def dart_matrix(path):
    d = pd.read_csv(path).set_index("layer")
    return d.values.astype(float)   # [layers, heads]


def fig_dart_heatmaps(out):
    panels = [("8b", DART/"results_llama8b/dart/toxic_heads_is_r64.csv"),
              ("70b", DART/"results_llama70b/dart/toxic_heads_is_r64.csv")]
    for tag, p in panels:
        if not Path(p).exists():
            continue
        fig, ax = plt.subplots(figsize=S.SIZE_1COL_SQUARE)
        M = dart_matrix(p)
        im = ax.imshow(M, aspect="auto", origin="lower", cmap=HEATMAP_CMAP)
        ax.grid(False)
        idx = np.dstack(np.unravel_index(np.argsort(M.ravel())[-5:][::-1], M.shape))[0]
        for (l, h) in idx:   # red squares mark the top-5 heads
            ax.scatter([h], [l], marker="s", facecolors="none", edgecolors=S.RED, s=40,
                       linewidths=0.8)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=.046, label="toxic top-50 count")
        fig.tight_layout()
        S.finalize(fig, out / f"fig_dart_heatmap_{tag}", png=True, sync=True)
        print(f"wrote fig_dart_heatmap_{tag}")


def fig_dart_layerprofile(out):
    fig, axes = plt.subplots(2, 1, figsize=S.SIZE_1COL_STACK, squeeze=False)
    for i, p in enumerate([DART/"results_llama8b/dart/toxic_heads_is_r64.csv",
                           DART/"results_llama70b/dart/toxic_heads_is_r64.csv"]):
        ax = axes[i][0]
        if not Path(p).exists(): ax.set_visible(False); continue
        M = dart_matrix(p); per = M.sum(1); nL = len(per); half = nL // 2
        colors = [S.ORANGE if l < half else S.BLUE for l in range(nL)]
        ax.bar(range(nL), per, color=colors)
        if i == 1:
            ax.set_xlabel("layer")
        ax.set_ylabel("summed toxic count")
        if i == 0:
            ax.legend(handles=[Patch(facecolor=S.ORANGE, label="early half"),
                               Patch(facecolor=S.BLUE, label="late half")],
                      loc="upper left")
    fig.tight_layout()
    S.finalize(fig, out / "fig_dart_layerprofile", png=True, sync=True)
    print("wrote fig_dart_layerprofile")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(S.FIGDIR))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig_injection_heatmaps(out)
    fig_injection_curves(out)
    fig_dart_heatmaps(out)
    fig_dart_layerprofile(out)
