#!/usr/bin/env python3
"""Seed stability of the 8B fine-grained lens claims (parity250 lenses,
seeds 0-2, identical 250-step annealed schedule).

DART: is the strongest head / the top-5 set / the ranking stable across
training seeds?  Injection: is the Delta_l site pick stable?

Seed-0 inputs are the existing parity250 analyses; seeds 1-2 come from
run_llama8b_seed_stability.pbs.  Writes seed_stability_{dart,injection}.csv
into data/seed_stability/ and prints the summary.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

TOX = Path("/path/to/project/interp/experiments/toxicity_ablation")
INJ = Path("/path/to/project/memory_injections/experiments/tweak_factor_analysis")
OUT = Path(__file__).resolve().parent.parent / "data" / "seed_stability"

LENSES = ["is", "topk", "tuned"]


def dart_csv(lens, seed):
    suf = "" if seed == 0 else f"_s{seed}"
    return TOX / f"results_llama8b/dart/toxic_heads_{lens}_parity250{suf}.csv"


def inj_dir(lens, seed):
    suf = "" if seed == 0 else f"_s{seed}"
    return INJ / f"results_llama8b_expanded/{lens}_parity250{suf}/2wmh"


def dart_stats(path):
    d = pd.read_csv(path).set_index("layer")
    M = d.values.astype(float)
    tot = M.sum() or 1.0
    flat = np.argsort(M.ravel())[::-1]
    top5 = [tuple(np.unravel_index(i, M.shape)) for i in flat[:5]]
    return {
        "top_head": f"L{top5[0][0]}.H{top5[0][1]}",
        "top1_share": 100 * M[top5[0]] / tot,
        "top5_share": 100 * sum(M[t] for t in top5) / tot,
        "top5_set": {f"L{l}.H{h}" for l, h in top5},
        "ranking": M.ravel(),
    }


def spearman(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def inj_pick(d):
    t0 = pd.read_csv(d / "tweak_0.csv")
    t1 = pd.read_csv(d / "tweak_1.csv")
    cols = [c for c in t0.columns if c.startswith("ans_prob_lens_edit_layer")]
    nL = len(cols)
    D = np.array([t1[f"ans_prob_lens_edit_layer{l}"].mean()
                  - t0[f"ans_prob_lens_edit_layer{l}"].mean() for l in range(nL)])
    order = np.argsort(D)[::-1]
    return int(order[0]), int(order[1]), D


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dart_rows, inj_rows = [], []
    for lens in LENSES:
        stats = {s: dart_stats(dart_csv(lens, s)) for s in (0, 1, 2)}
        for s, st in stats.items():
            dart_rows.append(dict(lens=lens, seed=s, top_head=st["top_head"],
                                  top1_share=round(st["top1_share"], 1),
                                  top5_share=round(st["top5_share"], 1),
                                  top5_set=";".join(sorted(st["top5_set"]))))
        heads = [st["top_head"] for st in stats.values()]
        ov = [len(stats[a]["top5_set"] & stats[b]["top5_set"])
              for a, b in combinations((0, 1, 2), 2)]
        sp = [spearman(stats[a]["ranking"], stats[b]["ranking"])
              for a, b in combinations((0, 1, 2), 2)]
        t1s = [round(stats[s]["top1_share"], 1) for s in (0, 1, 2)]
        t5s = [round(stats[s]["top5_share"], 1) for s in (0, 1, 2)]
        print(f"DART {lens:6s}: top head {heads}  "
              f"top5 overlap {ov}/5  spearman {[f'{x:.2f}' for x in sp]}  "
              f"top1 share {t1s}  top5 share {t5s}")

        picks = {}
        for s in (0, 1, 2):
            p1, p2, D = inj_pick(inj_dir(lens, s))
            picks[s] = (p1, p2)
            inj_rows.append(dict(lens=lens, seed=s, pick=p1, second=p2,
                                 delta_at_pick=round(float(D[p1]), 5)))
        print(f"INJ  {lens:6s}: Delta_l pick (1st/2nd) " +
              "  ".join(f"s{s}:L{p1}/L{p2}" for s, (p1, p2) in picks.items()))

    pd.DataFrame(dart_rows).to_csv(OUT / "seed_stability_dart.csv", index=False)
    pd.DataFrame(inj_rows).to_csv(OUT / "seed_stability_injection.csv", index=False)
    print("wrote", OUT / "seed_stability_dart.csv")
    print("wrote", OUT / "seed_stability_injection.csv")


if __name__ == "__main__":
    main()
