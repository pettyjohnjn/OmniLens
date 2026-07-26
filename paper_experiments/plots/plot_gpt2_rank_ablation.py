#!/usr/bin/env python3
"""GPT-2 LoRA rank ablation from the exact-KL evaluation results:

  rank_ablation_exact_kl_1col       two stacked KL panels vs rank
  rank_ablation_layerwise_heatmap   layerwise KL delta vs the baseline

Also writes rank_ablation_summary.csv next to the eval tree.
Data: evaluation/gpt2_preemptable_sweep_1000 aggregate_metrics.json."""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "style"))
import lens_style as S

import matplotlib.pyplot as plt

DEFAULT_EVAL_DIR = (Path(__file__).resolve().parents[1]
                    / "experiments/01_rank_ablation_estimator/data/evaluation"
                    / "gpt2_preemptable_sweep_1000")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Summary CSV location; defaults to "
                        "<eval-dir>/plots/rank_ablation_exact_kl.")
    return parser.parse_args()


def lens_kl_values(metrics):
    for key in ("tuned", "lora"):
        if key in metrics and "kl" in metrics[key]:
            kl = metrics[key]["kl"]
            values = []
            for layer in range(12):
                value = kl.get(f"layer_{layer}", kl.get(str(layer)))
                if value is None:
                    raise KeyError(f"Missing layer {layer} in KL metrics")
                values.append(float(value))
            return values
    raise KeyError(f"No tuned/lora KL metrics found. Keys: {sorted(metrics)}")


def load_metrics(path):
    with path.open() as handle:
        return json.load(handle)


def pearson(a, b):
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denom_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    denom_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    return numerator / (denom_a * denom_b)


def kendall_tau(a, b):
    concordant = 0
    discordant = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            product = (a[i] - a[j]) * (b[i] - b[j])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else float("nan")


def summarize(values):
    return {
        "final_layer_kl": values[-1],
        "mean_layer_kl": sum(values) / len(values),
        "early_layer_kl": sum(values[:6]) / 6.0,
    }


def collect(eval_dir, step):
    baseline_dir = eval_dir / f"tuned_kl_baseline__lens_step_{step}"
    baseline_values = lens_kl_values(load_metrics(baseline_dir / "aggregate_metrics.json"))
    baseline_summary = summarize(baseline_values)

    rows = []
    for path in sorted(eval_dir.glob(f"lora_kl_r*__lens_step_{step}/aggregate_metrics.json")):
        match = re.fullmatch(r"lora_kl_r(\d+)__lens_step_(\d+)", path.parent.name)
        if match is None:
            continue
        rank = int(match.group(1))
        values = lens_kl_values(load_metrics(path))
        summary = summarize(values)
        deviations = [abs(x - y) for x, y in zip(values, baseline_values)]
        rows.append({
            "rank": rank,
            "step": step,
            "layer_kl": values,
            "final_layer_kl": summary["final_layer_kl"],
            "mean_layer_kl": summary["mean_layer_kl"],
            "early_layer_kl": summary["early_layer_kl"],
            "mean_abs_dev_vs_fullrank": sum(deviations) / len(deviations),
            "max_dev_vs_fullrank": max(deviations),
            "pearson_vs_fullrank": pearson(values, baseline_values),
            "kendall_tau_vs_fullrank": kendall_tau(values, baseline_values),
        })

    rows.sort(key=lambda row: row["rank"])
    return baseline_summary, baseline_values, rows


def write_summary_csv(path, baseline_summary, rows):
    fieldnames = ["config", "rank", "step", "final_layer_kl", "mean_layer_kl",
                  "early_layer_kl", "mean_abs_dev_vs_fullrank",
                  "max_dev_vs_fullrank", "pearson_vs_fullrank",
                  "kendall_tau_vs_fullrank"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "config": "full_rank_tuned",
            "rank": "full",
            "step": rows[0]["step"] if rows else "",
            "final_layer_kl": baseline_summary["final_layer_kl"],
            "mean_layer_kl": baseline_summary["mean_layer_kl"],
            "early_layer_kl": baseline_summary["early_layer_kl"],
            "mean_abs_dev_vs_fullrank": 0.0,
            "max_dev_vs_fullrank": 0.0,
            "pearson_vs_fullrank": 1.0,
            "kendall_tau_vs_fullrank": 1.0,
        })
        for row in rows:
            out = {"config": f"lora_r{row['rank']}"}
            out.update({key: value for key, value in row.items() if key != "layer_kl"})
            writer.writerow(out)


def first_crossing_rank(ranks, values, baseline):
    for rank, value in zip(ranks, values):
        if value <= baseline:
            return rank
    return None


def add_baseline_fill(ax, ranks, values, baseline):
    ax.fill_between(ranks, [baseline] * len(ranks), values,
                    where=[value >= baseline for value in values],
                    color=S.RED, alpha=0.25, linewidth=0, interpolate=True)
    ax.fill_between(ranks, [baseline] * len(ranks), values,
                    where=[value <= baseline for value in values],
                    color=S.GREEN, alpha=0.25, linewidth=0, interpolate=True)


def annotate_crossing(ax, rank, baseline):
    if rank is None:
        return
    ax.axvline(rank, linestyle=":", linewidth=1.1, color=S.GREY)
    if rank >= 256:
        xytext, ha = (-50, 14), "right"
    else:
        xytext, ha = (8, 14), "left"
    ax.annotate("crosses\nbaseline", xy=(rank, baseline), xytext=xytext,
                textcoords="offset points", ha=ha, va="bottom",
                fontsize=S.FONT_ANNOT, color=S.DARK,
                arrowprops={"arrowstyle": "-", "linewidth": 0.8, "color": S.GREY})


def plot_1col(baseline_summary, rows):
    ranks = [row["rank"] for row in rows]
    final_kl = [row["final_layer_kl"] for row in rows]
    mean_kl = [row["mean_layer_kl"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(S.WIDTH_1COL, 3.5),
                             constrained_layout=True, sharex=True)
    panels = [
        (axes[0], final_kl, baseline_summary["final_layer_kl"], "final-layer KL"),
        (axes[1], mean_kl, baseline_summary["mean_layer_kl"], "mean layerwise KL"),
    ]
    for ax, ys, baseline, ylabel in panels:
        add_baseline_fill(ax, ranks, ys, baseline)
        ax.plot(ranks, ys, marker="o", markersize=3.5, color=S.BLUE)
        ax.axhline(baseline, linestyle="--", linewidth=1.2, color=S.TUNED_RED,
                   label=S.LENS_LABELS["tuned"])
        annotate_crossing(ax, first_crossing_rank(ranks, ys, baseline), baseline)
        ax.set_xscale("log", base=2)
        ax.set_xlim(ranks[0] * 0.85, ranks[-1] * 1.18)
        ax.set_xticks(ranks)
        # rotated so 256/384 do not collide on the log-2 axis
        ax.set_xticklabels([str(rank) for rank in ranks], rotation=35, ha="right")
        ax.minorticks_off()
        ax.set_ylabel(ylabel)
    axes[1].set_xlabel("LoRA rank")
    axes[0].legend(loc="upper right")

    return S.finalize(fig, S.FIGDIR / "rank_ablation_exact_kl_1col", sync=True)


def plot_heatmap(baseline_values, rows):
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    ranks = [row["rank"] for row in rows]
    layer_matrix = np.array([row["layer_kl"] for row in rows], dtype=float).T
    baseline = np.array(baseline_values, dtype=float).reshape(12, 1)
    improvement = baseline - layer_matrix
    # low-rank losses are much larger than high-rank gains; a signed log
    # transform keeps the near-baseline structure visible
    linthresh = 0.002
    signed_log_improvement = np.sign(improvement) * np.log10(
        1.0 + np.abs(improvement) / linthresh
    )

    def signed_log(value):
        return math.copysign(math.log10(1.0 + abs(value) / linthresh), value)

    cmap = plt.get_cmap("RdBu").copy()
    norm = TwoSlopeNorm(vmin=signed_log(-1.4), vcenter=0.0, vmax=signed_log(0.05))

    fig, ax = plt.subplots(figsize=(S.WIDTH_2COL, 3.0), constrained_layout=True)
    im = ax.imshow(signed_log_improvement, aspect="auto",
                   interpolation="nearest", cmap=cmap, norm=norm)
    ax.grid(False)
    ax.set_xticks(list(range(len(ranks))))
    ax.set_xticklabels([str(rank) for rank in ranks], rotation=35, ha="right")
    ax.set_yticks(list(range(12)))
    ax.set_yticklabels([str(layer) for layer in range(12)])
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("Layer")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025, extend="both")
    raw_ticks = [-1.0, -0.25, -0.10, -0.025, 0.0, 0.01, 0.025, 0.05]
    cbar.set_ticks([signed_log(value) for value in raw_ticks])
    cbar.set_ticklabels([f"{value:g}" for value in raw_ticks])
    cbar.set_label("Baseline KL - LoRA KL")
    cbar.ax.grid(False)

    return S.finalize(fig, S.FIGDIR / "rank_ablation_layerwise_heatmap", sync=True)


def main():
    args = parse_args()
    out_dir = args.out_dir or args.eval_dir / "plots" / "rank_ablation_exact_kl"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary, baseline_values, rows = collect(args.eval_dir, args.step)
    write_summary_csv(out_dir / "rank_ablation_summary.csv", baseline_summary, rows)

    pdf_1col = plot_1col(baseline_summary, rows)
    pdf_heatmap = plot_heatmap(baseline_values, rows)

    print(f"wrote {out_dir / 'rank_ablation_summary.csv'}")
    print(f"wrote {pdf_1col}")
    print(f"wrote {pdf_heatmap}")


if __name__ == "__main__":
    main()
