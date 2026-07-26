"""fig_allsites_ablation_bars: whole-model toxicity-ablation audit over the
expanded hookset vs the original attention-head scope, GPT-2/8B stack.
Data: 05_toxicity_dart_toxin/data (allhooks + toxin summaries, GPT-2
evaluation_900step trees)."""
import csv
import json

import numpy as np
import pandas as pd

from case_common import EXP, TOX, S, label, tag
import matplotlib.pyplot as plt

TOX900 = EXP / "05_toxicity_dart_toxin/data/evaluation_900step"


def cost_constrained(bt, bp, pts, thresh=1.10):
    """Best toxicity reduction among interventions within the PPL budget."""
    return max(((bt - t) / bt * 100 for _, t, p in pts if p / bp <= thresh),
               default=0.0)


def load_gpt2_allhooks(site_type, method="lora_r64"):
    rows = list(csv.DictReader(open(TOX900 / "toxicity_allhooks_ablation" / site_type / "summary.csv")))
    base = next(r for r in rows if r["method"] == "baseline")
    pts = sorted(
        [(float(r["memory_scale"]), float(r["mean_toxicity_score"]), float(r["perplexity"]))
         for r in rows if r["method"] == method],
        key=lambda x: x[0],
    )
    return float(base["mean_toxicity_score"]), float(base["perplexity"]), pts


def load_gpt2_perhead():
    matrix = TOX900 / "toxicity_ablation_matrix"
    pts = []
    for sd in sorted((matrix / "heads_lora" / "lora_r64").iterdir()):
        rj = sd / "result.json"
        if rj.exists():
            rd = json.load(open(rj))
            pts.append((float(rd["memory_scale"]),
                        float(rd["mean_toxicity_score"]),
                        float(rd["perplexity"])))
    pts.sort(key=lambda x: x[0])
    base_rows = list(csv.DictReader(open(matrix / "heads_lora" / "summary.csv")))
    base = next(r for r in base_rows if r["method"] == "baseline")
    return float(base["mean_toxicity_score"]), float(base["perplexity"]), pts


def load_8b_allhooks(st):
    f = TOX / "results_llama8b/allhooks" / st / "summary.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f)
    base = t[t.method == "baseline"].iloc[0]
    pts = [(r.memory_scale, r.mean_toxicity_score, r.perplexity)
           for r in t[t.method == "lora_r64"].itertuples()]
    return float(base.mean_toxicity_score), float(base.perplexity), pts


def load_8b_original_scope():
    t = pd.read_csv(TOX / "results_llama8b/toxin/ablation_summary.csv").set_index("config")
    bt, bp = t.loc["baseline", "mean_tox"], t.loc["baseline", "ppl"]
    pts = [(0.0, t.loc[c, "mean_tox"], t.loc[c, "ppl"]) for c in t.index
           if c != "baseline" and not c.startswith("zero_random")]
    return cost_constrained(bt, bp, pts)


def panel(ax, vals, ph, ph_label, tag_text):
    x = np.arange(len(S.SITE_TYPES))
    bars = ax.bar(x, vals, 0.62, color=S.ORANGE, label="expanded hookset")
    b_ph = ax.bar([len(S.SITE_TYPES)], [ph], 0.62, color=S.HEUR_FACE, hatch="//",
                  edgecolor="white", linewidth=0, label=ph_label)
    label(ax, list(bars) + list(b_ph), fmt="{:+.1f}")
    ax.axhline(ph, color=S.GREY, lw=1.0, ls="--")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(x) + [len(S.SITE_TYPES)],
                  list(S.SITE_TYPES) + ["attn heads\n(original)"],
                  fontsize=S.FONT_ANNOT, rotation=30, ha="right")
    ax.set_ylabel("toxicity reduction (%)")
    tag(ax, tag_text)
    # headroom so the tag and bar labels never collide
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.18 * (hi - lo))


def main():
    gpt2_vals = [cost_constrained(*load_gpt2_allhooks(st)) for st in S.SITE_TYPES]
    gpt2_ph = cost_constrained(*load_gpt2_perhead())
    l8b = [load_8b_allhooks(st) for st in S.SITE_TYPES]

    if not all(x is not None for x in l8b):
        fig, ax = plt.subplots(figsize=S.SIZE_1COL, constrained_layout=True)
        panel(ax, gpt2_vals, gpt2_ph, "original scope (attn heads only)", "GPT-2")
        ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    else:
        vals8 = [cost_constrained(*x) for x in l8b]
        ph8 = load_8b_original_scope()
        fig, axes = plt.subplots(2, 1, figsize=S.SIZE_1COL_STACK,
                                 constrained_layout=True)
        panel(axes[0], gpt2_vals, gpt2_ph, "original scope (attn heads)",
              "GPT-2 (12 layers)")
        panel(axes[1], vals8, ph8, "original scope (attn heads)",
              "LLaMA-3-8B (32 layers)")
        axes[1].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.55),
                       fontsize=S.FONT_ANNOT)
    S.finalize(fig, S.FIGDIR / "fig_allsites_ablation_bars.pdf", sync=True)
    print("wrote fig_allsites_ablation_bars")


if __name__ == "__main__":
    main()
