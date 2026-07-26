#!/usr/bin/env python3
"""top_token_agreement_heatmap: top-1 agreement / top-10 overlap vs the
full-rank baseline, layer x checkpoint. Data: layerwise_top_token_agreement.csv
(read verbatim; the GPU compute path is not rerun)."""
import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "style"))
import lens_style as S

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CSV = (Path(__file__).resolve().parents[1]
               / "experiments/01_rank_ablation_estimator/data/evaluation"
               / "gpt2_preemptable_sweep_1000/plots/top_token_agreement"
               / "layerwise_top_token_agreement.csv")
DEFAULT_OUT = S.FIGDIR / "top_token_agreement_heatmap.pdf"


def load_matrices(csv_path):
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    topk_name = next(name for name in fieldnames if re.fullmatch(r"top\d+_overlap", name))
    top_k = int(re.fullmatch(r"top(\d+)_overlap", topk_name).group(1))

    header = [row for row in rows if row["layer"] == "0"]
    header.sort(key=lambda r: (r.get("family", ""), r["run_name"]))
    labels = [r["config_label"] for r in header]
    run_names = [r["run_name"] for r in header]
    families = [r.get("family", "") for r in header]
    layers = sorted({int(row["layer"]) for row in rows})
    by_key = {(int(row["layer"]), row["run_name"]): row for row in rows}

    top1 = np.zeros((len(layers), len(labels)))
    topk = np.zeros_like(top1)
    for y, layer in enumerate(layers):
        for x, run_name in enumerate(run_names):
            row = by_key[(layer, run_name)]
            top1[y, x] = float(row["top1_agreement"])
            topk[y, x] = float(row[topk_name])
    return labels, families, layers, top1, topk, top_k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    labels, families, layers, top1, topk, top_k = load_matrices(args.csv)
    print(f"{len(labels)} checkpoints x {len(layers)} layers "
          f"(top1 mean {top1.mean():.4f}, top{top_k} mean {topk.mean():.4f})")

    # 99 per-checkpoint labels are unreadable at this width, so the x-axis
    # is labeled by run family with thin separators between groups
    bounds, centers, names = [], [], []
    start = 0
    for i in range(1, len(families) + 1):
        if i == len(families) or families[i] != families[start]:
            centers.append((start + i - 1) / 2)
            names.append(families[start].replace("_", " "))
            if i < len(families):
                bounds.append(i - 0.5)
            start = i

    fig, axes = plt.subplots(1, 2, figsize=S.SIZE_2COL_TALL, constrained_layout=True)
    panels = [
        (axes[0], top1, "Top-1 agreement"),
        (axes[1], topk, f"Top-{top_k} overlap"),
    ]
    for ax, matrix, panel_label in panels:
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest",
                       cmap="viridis", vmin=0.0, vmax=1.0)
        ax.grid(False)
        ax.set_xlabel("Checkpoint (grouped by run family)")
        ax.set_ylabel("Layer")
        for b in bounds:
            ax.axvline(b, color="white", linewidth=0.6, alpha=0.8)
        ax.set_xticks(centers)
        ax.set_xticklabels(names, fontsize=S.FONT_ANNOT,
                           rotation=30, ha="right")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(layer) for layer in layers])
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label(panel_label)

    pdf = S.finalize(fig, args.out, png=True, sync=True)
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
