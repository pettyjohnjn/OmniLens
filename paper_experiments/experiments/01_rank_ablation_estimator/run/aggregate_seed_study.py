#!/usr/bin/env python3
"""Aggregate the GPT-2 seed study + RS baseline into per-seed and mean+-sd
tables for the paper (Table 2 RS rows, seed-spread statements).

All KL numbers come from the identical Belrose eval loop (131,072 Pile test
tokens): Final = layer 11, Mean = mean over the 12 residual layers, Early =
mean over layers 0-3 -- the same reductions as every existing Table-1/2 cell.

Seed 0 of the four replicated recipes is read from the original sweep eval
trees (the exact dirs the paper's current numbers come from); seeds 1-2 and
all RS runs come from evaluation/gpt2_seed_study_1000 (eval_gpt2_seed_study.pbs).

Writes seed_study_per_run.csv and seed_study_summary.csv next to the other
canonical CSVs in data/evaluation/seed_study/ and prints the summary.
"""
import json
import statistics as st
from pathlib import Path

LT = Path("/path/to/project/eval_harness/evaluation")
LR = Path("/path/to/project/omnilens/evaluation")
SEED = LR / "gpt2_seed_study_1000"
OUT = Path("/path/to/project/paper_experiments/experiments/"
           "01_rank_ablation_estimator/data/evaluation/seed_study")

# config -> {seed: eval dir}
RUNS = {
    "tuned_fullkl": {
        0: LT / "gpt2_preemptable_sweep_1000/tuned_kl_baseline__lens_step_1000",
        1: SEED / "tuned_kl_seed1__lens_step_1000",
        2: SEED / "tuned_kl_seed2__lens_step_1000",
    },
    "lora_fullkl": {
        0: LT / "gpt2_preemptable_sweep_1000/lora_kl_r64__lens_step_1000",
        1: SEED / "lora_kl_r64_seed1__lens_step_1000",
        2: SEED / "lora_kl_r64_seed2__lens_step_1000",
    },
    "topk_k1024": {
        0: LT / "gpt2_preemptable_sweep_1000/lora_subset_topk_r64_k1024__lens_step_1000",
        1: SEED / "lora_subset_topk_r64_k1024_seed1__lens_step_1000",
        2: SEED / "lora_subset_topk_r64_k1024_seed2__lens_step_1000",
    },
    "is_512_512": {
        0: LR / "gpt2_is_sweeps_and_expanded_final/"
              "gpt2_preemptable_is_sweep_1000__lora_subset_is_r64_k512_tail512__lens_step_1000",
        1: SEED / "lora_subset_is_r64_k512_tail512_seed1__lens_step_1000",
        2: SEED / "lora_subset_is_r64_k512_tail512_seed2__lens_step_1000",
    },
    "rs_tail512": {s: SEED / f"lora_subset_rs_r64_tail512_seed{s}__lens_step_1000"
                   for s in (0, 1, 2)},
    "rs_tail1024": {s: SEED / f"lora_subset_rs_r64_tail1024_seed{s}__lens_step_1000"
                    for s in (0, 1, 2)},
    "rs_tail1536": {0: SEED / "lora_subset_rs_r64_tail1536_seed0__lens_step_1000"},
}


def layer_key(name):
    return int(name) if str(name).isdigit() else int(str(name).rsplit("_", 1)[1])


def metrics(eval_dir):
    data = json.loads((eval_dir / "aggregate_metrics.json").read_text())
    block = data.get("tuned") or data.get("lora")
    kl = [float(v) for _, v in sorted(block["kl"].items(), key=lambda kv: layer_key(kv[0]))]
    assert len(kl) == 12, f"{eval_dir}: {len(kl)} layers"
    return kl[-1], st.mean(kl), st.mean(kl[:4])


def fmt(x, sd=None, p=4):
    return f"{x:.{p}f}" + (f" +- {sd:.{p}f}" if sd is not None else "")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_rows = [("config", "seed", "final_kl", "mean_kl", "early_kl", "eval_dir")]
    sum_rows = [("config", "n_seeds", "final_mean", "final_sd",
                 "mean_mean", "mean_sd", "early_mean", "early_sd")]
    print(f"{'config':14s} {'final KL':>22s} {'mean KL':>22s} {'early KL':>22s}")
    for cfg, seeds in RUNS.items():
        vals = {}
        for s, d in sorted(seeds.items()):
            f, m, e = metrics(d)
            vals[s] = (f, m, e)
            per_rows.append((cfg, s, f"{f:.6f}", f"{m:.6f}", f"{e:.6f}", str(d)))
        fs, ms, es = zip(*vals.values())
        sd = (lambda xs: st.stdev(xs) if len(xs) > 1 else float("nan"))
        sum_rows.append((cfg, len(vals), f"{st.mean(fs):.6f}", f"{sd(fs):.6f}",
                         f"{st.mean(ms):.6f}", f"{sd(ms):.6f}",
                         f"{st.mean(es):.6f}", f"{sd(es):.6f}"))
        print(f"{cfg:14s} {fmt(st.mean(fs), sd(fs) if len(fs)>1 else None):>22s}"
              f" {fmt(st.mean(ms), sd(ms) if len(ms)>1 else None, 3):>22s}"
              f" {fmt(st.mean(es), sd(es) if len(es)>1 else None, 3):>22s}"
              f"   (seeds: {sorted(vals)})")
        for s in sorted(vals):
            f, m, e = vals[s]
            print(f"    seed{s}: final {f:.4f}  mean {m:.3f}  early {e:.3f}")
    for name, rows in [("seed_study_per_run.csv", per_rows),
                       ("seed_study_summary.csv", sum_rows)]:
        (OUT / name).write_text("\n".join(",".join(map(str, r)) for r in rows) + "\n")
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
