"""Shared paths, loaders, and bar-chart helpers for the case-study figures."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "style"))
import lens_style as S

EXP = Path(__file__).resolve().parents[1] / "experiments"
INJ = EXP / "04_memory_injection/data"
TOX = EXP / "05_toxicity_dart_toxin/data"
TRAJ = EXP / "03_injection_detection/data"

MODELS = {"GPT-2": "results_gpt2_expanded",
          "LLaMA-3-8B": "results_llama8b_expanded",
          "LLaMA-3-70B": "results_llama70b_expanded"}


def tag(ax, text, x=0.02, y=0.98):
    """Short in-axes panel label (figures carry no top titles)."""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top")


def label(ax, bars, fmt="{:.1f}", fs=S.FONT_ANNOT):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h),
                ha="center", va="bottom" if h >= 0 else "top", fontsize=fs)


def load_summary():
    """Injection-summary rows (2wmh, non-null). 8B picks come from the
    parity250 checkpoints; 70B/GPT-2 keep their single-schedule names."""
    df = pd.read_csv(INJ / "alllens_injection_summary_parity250.csv")
    return df[(df.ds == "2wmh") & (~df.null)]


# Each chooser lists every tag it may carry across models; at most one
# tag exists per model (8B rows use parity250 names, 70B keeps is_r64).
CHOOSERS = [(("lens:is_r64", "lens:is_parity250"), "OmniLens Top-$k$+IS", S.ORANGE, None),
            (("lens:topk_r64", "lens:topk_parity250"), "OmniLens Top-$k$", S.BLUE, None),
            (("lens:logit",), "logit lens", S.GREY, None),
            (("heuristic(last)",), "final-layer selector", S.HEUR_FACE, "//")]
GROUPS = [("LLaMA-3-8B", "2.0"), ("LLaMA-3-8B", "4.0"),
          ("LLaMA-3-70B", "2.0"), ("LLaMA-3-70B", "4.0")]
GROUP_LABELS = ["8B\n$\\tau{=}2$", "8B\n$\\tau{=}4$",
                "70B\n$\\tau{=}2$", "70B\n$\\tau{=}4$"]


def grouped(ax, df, col, scale=1.0, fmt="{:.0f}", choosers=CHOOSERS, labels=True,
            groups=GROUPS, group_labels=GROUP_LABELS):
    """Grouped bars: x = (model, tau) groups, one bar per chooser.
    Missing rows simply omit the bar."""
    x = np.arange(len(groups))
    w = 0.8 / len(choosers)
    off = (len(choosers) - 1) / 2
    for k, (ch, lbl, color, hatch) in enumerate(choosers):
        tags = (ch,) if isinstance(ch, str) else ch
        vals, pos = [], []
        for g, (m, tau) in enumerate(groups):
            r = df[(df.model == m) & (df.tau == tau) & (df.chooser.isin(tags))]
            if len(r):
                vals.append(float(r[col].iloc[0]) * scale)
                pos.append(x[g] + (k - off) * w)
        bars = ax.bar(pos, vals, w, label=lbl, color=color, hatch=hatch,
                      edgecolor="white" if hatch else None, linewidth=0)
        if labels:
            label(ax, bars, fmt=fmt)
    ax.set_xticks(x, group_labels)
