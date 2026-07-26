"""Shared Matplotlib style for the paper figures.

Importing this module applies the rcParams (Agg backend, 9pt base fonts,
no bold). Save every figure through finalize(), which strips titles,
writes pdf (+png), and with sync=True copies the pdf into the paper trees.

    import sys; sys.path.insert(0, "<repo>/style")
    import lens_style as S

    fig, ax = plt.subplots(figsize=S.SIZE_1COL)
    ax.plot(x, y, color=S.ORANGE)
    S.finalize(fig, S.FIGDIR / "my_fig", sync=True)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
# Optional extra directories that finalize(sync=True) mirrors figures into
# (used to keep the paper's own figures/ tree in sync while writing).
PAPER_TREES = []

# Figure footprints (inches). Widths are fixed so on-page text is uniform
# when each figure is included at its native width (two-column paper format).
WIDTH_1COL = 3.4
WIDTH_2COL = 7.0
SIZE_1COL = (WIDTH_1COL, 2.4)
SIZE_1COL_SQUARE = (WIDTH_1COL, 3.0)
SIZE_1COL_STACK = (WIDTH_1COL, 4.6)
SIZE_2COL = (WIDTH_2COL, 2.7)
SIZE_2COL_TALL = (WIDTH_2COL, 3.6)

FONT_BASE = 9
FONT_SMALL = 8
FONT_TICK = 8
FONT_ANNOT = 7

DARK = "#222222"
GREY = "#607D8B"
BLUE = "#4878CF"
ORANGE = "#FF7F0E"
GREEN = "#6ACC65"
RED = "#D65F5F"
PURPLE = "#B47CC7"
GOLD = "#C4AD66"
TUNED_RED = "#E53935"     # full-rank tuned lens
HEUR_FACE = "#CFD8DC"     # hatched grey: free depth heuristic

SITE_TYPES = ["attn_in", "attn_out", "resid_mid", "mlp_in", "mlp_out", "resid_post"]
SITE_COLORS = {
    "attn_in": BLUE,
    "attn_out": ORANGE,
    "resid_mid": GREEN,
    "mlp_in": RED,
    "mlp_out": PURPLE,
    "resid_post": GOLD,
}

LENS_COLORS = {
    "is": ORANGE,          # low-rank Top-k+IS
    "topk": BLUE,          # LoRA Top-k
    "tuned": TUNED_RED,    # full-rank reference
    "logit": GREY,
    "unembed": GREY,
}
LENS_LABELS = {
    "is": "OmniLens Top-$k$+IS",
    "topk": "OmniLens Top-$k$",
    "tuned": "full-rank ref.",
    "logit": "logit lens",
}

_RC = {
    "font.family": "sans-serif",
    "font.size": FONT_BASE,
    "font.weight": "normal",
    "axes.titlesize": FONT_BASE,
    "axes.titleweight": "normal",
    "axes.labelsize": FONT_BASE,
    "axes.labelweight": "normal",
    "figure.titlesize": FONT_BASE,
    "figure.titleweight": "normal",
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_SMALL,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.7,
    "lines.linewidth": 1.6,
    "patch.linewidth": 0.0,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # embed TrueType, no Type-3
    "ps.fonttype": 42,
}


def apply_style():
    plt.rcParams.update(_RC)


apply_style()


def despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def strip_titles(fig):
    """No figure carries a top title; finalize() enforces this on save."""
    for ax in fig.get_axes():
        for loc in ("center", "left", "right"):
            if ax.get_title(loc):
                ax.set_title("", loc=loc)
    sup = getattr(fig, "_suptitle", None)
    if sup is not None:
        sup.set_text("")


def finalize(fig, path, png=True, sync=False):
    """Strip titles, write path.pdf (+ .png), close, optionally sync."""
    strip_titles(fig)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf)
    if png:
        fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    if sync:
        sync_figures([pdf])
    return pdf


def sync_figures(pdf_paths):
    import shutil

    for pdf in pdf_paths:
        pdf = Path(pdf)
        for tree in PAPER_TREES:
            if not tree.exists():
                continue
            try:
                shutil.copy2(pdf, tree / pdf.name)
            except OSError as e:
                print(f"[lens_style] sync to {tree} skipped: {e}")
