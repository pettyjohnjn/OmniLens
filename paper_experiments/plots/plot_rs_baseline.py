#!/usr/bin/env python3
"""fig_rs_layerwise: layerwise KL of the RS (pure teacher sampling) baseline
against the subset estimators at matched budgets. Curves are seed 0; shaded
bands span min-max over the three seeds where available.
Data: gpt2_seed_study_1000 + the original sweep evaluation trees."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "style"))
import lens_style as S

import matplotlib.pyplot as plt

EVAL = Path(__file__).resolve().parents[1] / "experiments/01_rank_ablation_estimator/data/evaluation"
SEED = EVAL / "gpt2_seed_study_1000"
LT = EVAL
LR = EVAL

# label, color, linestyle, eval dirs (seed 0 first)
SPECS = [
    ("RS $0{+}512$", S.GREY, ":",
     [SEED / f"lora_subset_rs_r64_tail512_seed{s}__lens_step_1000" for s in (0, 1, 2)]),
    ("RS $0{+}1024$", S.GREY, "--",
     [SEED / f"lora_subset_rs_r64_tail1024_seed{s}__lens_step_1000" for s in (0, 1, 2)]),
    ("RS $0{+}1536$", S.GREY, "-",
     [SEED / "lora_subset_rs_r64_tail1536_seed0__lens_step_1000"]),
    ("Top-$k$ 1024", S.LENS_COLORS["topk"], "-",
     [LT / "gpt2_preemptable_sweep_1000/lora_subset_topk_r64_k1024__lens_step_1000",
      SEED / "lora_subset_topk_r64_k1024_seed1__lens_step_1000",
      SEED / "lora_subset_topk_r64_k1024_seed2__lens_step_1000"]),
    ("Top-$k$+IS 512+512", S.LENS_COLORS["is"], "-",
     [LR / "gpt2_is_sweeps_and_expanded_final/"
           "gpt2_preemptable_is_sweep_1000__lora_subset_is_r64_k512_tail512__lens_step_1000",
      SEED / "lora_subset_is_r64_k512_tail512_seed1__lens_step_1000",
      SEED / "lora_subset_is_r64_k512_tail512_seed2__lens_step_1000"]),
]


def layer_key(name):
    return int(name) if str(name).isdigit() else int(str(name).rsplit("_", 1)[1])


def layers_of(eval_dir):
    data = json.loads((eval_dir / "aggregate_metrics.json").read_text())
    block = data.get("tuned") or data.get("lora")
    return [float(v) for _, v in sorted(block["kl"].items(), key=lambda kv: layer_key(kv[0]))]


def main():
    fig, ax = plt.subplots(figsize=S.SIZE_1COL, constrained_layout=True)
    for label, color, ls, dirs in SPECS:
        seeds = [layers_of(d) for d in dirs]
        base = seeds[0]
        ax.plot(range(len(base)), base, ls, color=color, lw=1.7, label=label)
        if len(seeds) > 1:
            lo = [min(v) for v in zip(*seeds)]
            hi = [max(v) for v in zip(*seeds)]
            ax.fill_between(range(len(base)), lo, hi, color=color, alpha=0.16,
                            linewidth=0, zorder=1)
        print(f"{label}: final {base[-1]:.3f} ({len(seeds)} seed(s))")
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL divergence")
    ax.set_xlim(-0.2, 11.2)
    ax.set_xticks(range(12))
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              fontsize=S.FONT_ANNOT)
    S.finalize(fig, S.FIGDIR / "fig_rs_layerwise.pdf", png=True, sync=True)
    print("wrote fig_rs_layerwise")


if __name__ == "__main__":
    main()
