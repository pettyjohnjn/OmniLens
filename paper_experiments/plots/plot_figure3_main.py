#!/usr/bin/env python3
"""figure3_main: layerwise KL for the budget-matched estimator comparison.
Five curves: full-KL reference plus (top-k, Top-k+IS) at total budgets 512
(dashed) and 1024 (solid); dotted vertical marker at each matched-budget
crossover. Seed bands span min-max over three seeds where available."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "style"))
import lens_style as S

import matplotlib.pyplot as plt

EVAL = Path(__file__).resolve().parents[1] / "experiments/01_rank_ablation_estimator/data/evaluation"
EVAL_ROOTS = [
    EVAL / "gpt2_debug_lowrank_sweep_1000",
    EVAL / "gpt2_preemptable_sweep_1000",
    EVAL / "gpt2_is_sweeps_and_expanded_final",
]
SEED_ROOT = EVAL / "gpt2_seed_study_1000"
DEFAULT_OUT = S.FIGDIR / "figure3_main.pdf"

MC_PREFIX = "gpt2_preemptable_is_sweep_1000__"

# run_name, label, color, linestyle, linewidth
SPECS = [
    ("lora_kl_r64", "Full-KL (ref.)", S.DARK, "-", 2.2),
    ("lora_subset_topk_r64_k512", "Top-$k$ 512", S.LENS_COLORS["topk"], "--", 1.5),
    (MC_PREFIX + "lora_subset_is_r64_k256_tail256", "Top-$k$+IS 256+256", S.LENS_COLORS["is"], "--", 1.5),
    ("lora_subset_topk_r64_k1024", "Top-$k$ 1024", S.LENS_COLORS["topk"], "-", 1.9),
    (MC_PREFIX + "lora_subset_is_r64_k512_tail512", "Top-$k$+IS 512+512", S.LENS_COLORS["is"], "-", 1.9),
]

# run_name -> seed-study eval names (seeds 1-2); the plotted line stays seed 0
SEED_RUNS = {
    "lora_kl_r64": ["lora_kl_r64_seed1", "lora_kl_r64_seed2"],
    "lora_subset_topk_r64_k1024": ["lora_subset_topk_r64_k1024_seed1",
                                   "lora_subset_topk_r64_k1024_seed2"],
    MC_PREFIX + "lora_subset_is_r64_k512_tail512": [
        "lora_subset_is_r64_k512_tail512_seed1",
        "lora_subset_is_r64_k512_tail512_seed2"],
}


def layer_key(name):
    return int(name) if str(name).isdigit() else int(str(name).rsplit("_", 1)[1])


def seed_layers(name):
    path = SEED_ROOT / f"{name}__lens_step_1000" / "aggregate_metrics.json"
    data = json.loads(path.read_text())
    block = data.get("tuned") or data.get("lora")
    return [float(v) for _, v in sorted(block["kl"].items(), key=lambda kv: layer_key(kv[0]))]


def best_layers(run_name, max_step=1000):
    best = None
    for root in EVAL_ROOTS:
        for path in root.glob(f"{run_name}__lens_step_*/aggregate_metrics.json"):
            step = int(path.parent.name.rsplit("__lens_step_", 1)[1])
            if step > max_step:
                continue
            data = json.loads(path.read_text())
            block = data.get("tuned") or data.get("lora")
            if block is None or "kl" not in block:
                continue
            layers = [float(v) for _, v in sorted(block["kl"].items(), key=lambda kv: layer_key(kv[0]))]
            if best is None or step > best[0]:
                best = (step, layers)
    if best is None:
        raise FileNotFoundError(f"no aggregate_metrics.json found for {run_name}")
    return best


def crossover_layer(topk, head_is):
    """Last layer at which the IS curve is below top-k before top-k takes over."""
    diffs = [t - m for t, m in zip(topk, head_is)]
    for layer in range(len(diffs) - 1):
        if diffs[layer] > 0 and diffs[layer + 1] <= 0:
            return layer + 0.5
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    curves = {}
    fig, ax = plt.subplots(figsize=S.SIZE_1COL, constrained_layout=True)
    ax.axvspan(-0.5, 3.5, color="#f0f0f0", zorder=0)   # early-layer band
    for run_name, label, color, linestyle, linewidth in SPECS:
        step, layers = best_layers(run_name)
        curves[run_name] = layers
        ax.plot(range(len(layers)), layers, label=label, color=color,
                linestyle=linestyle, linewidth=linewidth)
        print(f"{label}: step {step}, final KL {layers[-1]:.4f}")
        if run_name in SEED_RUNS:
            seeds = [layers] + [seed_layers(n) for n in SEED_RUNS[run_name]]
            lo = [min(vals) for vals in zip(*seeds)]
            hi = [max(vals) for vals in zip(*seeds)]
            ax.fill_between(range(len(layers)), lo, hi, color=color, alpha=0.16,
                            linewidth=0, zorder=1)
            print(f"  seed band: final KL {min(s[-1] for s in seeds):.4f}"
                  f"-{max(s[-1] for s in seeds):.4f} over {len(seeds)} seeds")

    for topk_run, mc_run, budget in [
        ("lora_subset_topk_r64_k512", MC_PREFIX + "lora_subset_is_r64_k256_tail256", 512),
        ("lora_subset_topk_r64_k1024", MC_PREFIX + "lora_subset_is_r64_k512_tail512", 1024),
    ]:
        layer = crossover_layer(curves[topk_run], curves[mc_run])
        if layer is not None:
            ax.axvline(layer, linestyle=":", linewidth=1.0, color="#777777")
            print(f"crossover at layer {layer} for budget {budget}")

    ax.set_xlabel("Layer")
    ax.set_ylabel("KL divergence")
    ax.set_xlim(-0.2, 11.2)
    ax.set_xticks(range(12))
    # opaque patch so the crossover lines do not run through the legend text
    ax.legend(loc="upper right", fontsize=S.FONT_ANNOT, frameon=True,
              framealpha=1.0, facecolor="white", edgecolor="none")

    pdf = S.finalize(fig, args.out, png=True, sync=True)
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
