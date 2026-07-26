#!/usr/bin/env python3
"""layerwise_pearson_kendall_heatmap: mean Pearson r / Kendall tau@100 vs the
full-rank baseline, layer x LoRA rank. Appendix companion of
top_token_agreement_heatmap. Data: layerwise_pearson_kendall.csv (read
verbatim; no compute is rerun)."""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "style"))
import lens_style as S

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CSV = (Path(__file__).resolve().parents[1]
               / "experiments/01_rank_ablation_estimator/data/evaluation"
               / "gpt2_preemptable_sweep_1000/plots/pearson_kendall_token_level"
               / "layerwise_pearson_kendall.csv")
DEFAULT_OUT = S.FIGDIR / "layerwise_pearson_kendall_heatmap.pdf"

REQUIRED_COLUMNS = {"rank", "layer", "mean_pearson", "mean_kendall_top100"}


def load_matrices(csv_path):
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    missing = REQUIRED_COLUMNS - fieldnames
    if missing:
        raise SystemExit(f"{csv_path} is missing columns: {sorted(missing)}")

    ranks = sorted({int(row["rank"]) for row in rows})
    layers = sorted({int(row["layer"]) for row in rows})
    by_key = {(int(row["layer"]), int(row["rank"])): row for row in rows}

    pearson = np.zeros((len(layers), len(ranks)))
    kendall = np.zeros_like(pearson)
    for y, layer in enumerate(layers):
        for x, rank in enumerate(ranks):
            row = by_key[(layer, rank)]
            pearson[y, x] = float(row["mean_pearson"])
            kendall[y, x] = float(row["mean_kendall_top100"])
    return ranks, layers, pearson, kendall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    ranks, layers, pearson, kendall = load_matrices(args.csv)
    print(f"{len(ranks)} ranks x {len(layers)} layers "
          f"(pearson mean {pearson.mean():.4f}, kendall mean {kendall.mean():.4f})")

    fig, axes = plt.subplots(1, 2, figsize=S.SIZE_2COL_TALL, constrained_layout=True)
    # Kendall tau@100 goes negative, so its panel autoscales instead of
    # clipping at 0; panel identity is carried by the colorbar label
    panels = [
        (axes[0], pearson, "Pearson $r$", 0.0, 1.0),
        (axes[1], kendall, r"Kendall $\tau$@100", None, None),
    ]
    for ax, matrix, panel_label, vmin, vmax in panels:
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest",
                       cmap="viridis", vmin=vmin, vmax=vmax)
        ax.grid(False)
        ax.set_xlabel("LoRA rank")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels([str(rank) for rank in ranks], fontsize=S.FONT_ANNOT)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(layer) for layer in layers], fontsize=S.FONT_ANNOT)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label(panel_label)
        cbar.ax.tick_params(labelsize=S.FONT_ANNOT)

    pdf = S.finalize(fig, args.out, png=True, sync=True)
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
