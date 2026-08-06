# 01 — GPT-2 Rank Ablation + Estimator Comparison

Consolidated experiment family: the GPT-2 rank x objective lens grid (rank ablation,
Table 1) and the subset-KL estimator comparison at r=64 (Table 2), plus the
token-level correlation analyses (appendix heatmaps). All files here were copied
read-only from `eval_harness/` and `omnilens/` under
`/path/to/project/`; sources were never modified.

## 1. The two stages

**Stage 1 — train the lens grid.** The rank x objective grid (low-rank lenses at
r ∈ {1,4,8,16,32,64,128,256,384}, objectives: full KL, subset-KL top-k, head/tail
and mixed-tail variants, plus the full-rank tuned-lens KL baseline) is trained
with the **omnilens trainer** in `omnilens/` (upstream repo — **not
copied here**; see section 4). Submit manifests in `run/jobs/submit/` define each
sweep; `run/jobs/pbs/run_sweep_manifest.sh` is the generic PBS payload.

**Stage 2 — evaluate.** Checkpoints are translated to tuned-lens format and
evaluated with the **forked tuned_lens harness** in `run/tuned_lens/`. This is a
load-bearing local fork (custom `scripts/eval_loop.py`, `load_artifacts.py`,
omnilens translation, exact-KL bookkeeping) — it must be on `PYTHONPATH` ahead of
any pip-installed tuned-lens; do not pip-install upstream tuned-lens over it.
Correlation scripts (`compute_gpt2_pearson_kendall_token_level.py`,
`compute_gpt2_top_token_agreement.py`, `merge_pearson_kendall_shards.py`) run over
the eval outputs to produce the token-level agreement CSVs.

## 2. How to run

Training (the cluster PBS, conda env `omnilens` — activated by `jobs/lib/common.sh`):

    cd run/jobs/submit
    ./gpt2_lowrank_sweep_1.sh          # parts 1-4 wrap submit_gpt2_lowrank_sweep_part.sh
    ./gpt2_r64_estimator_comparison_250_debug.sh
    # resume variants: gpt2_lowrank_sweep_resume_all{,_prod,_capacity,_preemptable}.sh
    # head+IS sweeps: gpt2_{standard,lowrank}_head_is_sweep_resume_all_prod.sh
    # scaling-debug sweeps: gpt2_{mixed_tail_proposal,subset_k2_k3_lr}_sweep_250_debug.sh

Each manifest sets env vars and qsubs `run/jobs/pbs/run_sweep_manifest.sh`, which
walks up parent dirs to find `jobs/lib/common.sh` — the `run/jobs/{pbs,lib,submit}`
layout preserved here keeps that resolution working (repo root resolves to `run/`).
Seed study: `run/gpt2_seed_study.pbs` + `run/gpt2_seed_worker.sh`.

Evaluation (conda env `tuned_lens_env`; single node, 4 GPUs):

    qsub run/eval.pbs        # SRC_ROOT/OUT_ROOT/EVAL_STEPS overridable via -v
    # core invocation inside:
    torchrun --standalone --nnodes=1 --nproc_per_node=4 -m tuned_lens eval ...

`eval.pbs` sets `PYTHONPATH=eval_harness:omnilens/src:subset-kl/src`
so the fork in `run/tuned_lens/` shadows any installed tuned-lens. Set
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for any local
(non-PBS) python runs.

Correlations (same env):

    qsub run/compute_pearson_kendall.pbs               # single-node token-level Pearson/Kendall
    qsub run/compute_pearson_kendall_by_rank_9node.pbs # 9-node by-rank sharding
    #   workers: run/run_pearson_kendall_rank_worker.sh, run/run_pearson_kendall_rank_mpi_worker.sh
    #   shard merge: run/merge_pearson_kendall_shards.py
    qsub run/top_token_agreement_gpt2.pbs              # top-1 / top-10 token agreement

Note: the `section4_*.csv` aggregates in `data/evaluation/section4_figures/` are
**written by the plotting stage** (`eval_harness/plot_section4_subsetkl_figures.py`,
not copied — plotting scripts live at the `eval_harness` root), which aggregates
the eval trees below into the paper's Section-4 figures.

## 3. Data map (what feeds which paper asset)

`data/evaluation/` preserves each source tree's relative structure; only
`aggregate_metrics.json` (+ `batches.jsonl`, all <100KB) per
`<run>__lens_step_*/` directory was copied — no checkpoints, no logs.

| data/ subtree | source root | feeds |
|---|---|---|
| `gpt2_preemptable_sweep_1000/` (447 runs) | `eval_harness/evaluation/` | **Table 1** (rank ablation), figures `rank_ablation_exact_kl_1col` + `rank_ablation_layerwise_heatmap`, `figure3_main` |
| `gpt2_preemptable_lowrank_is_sweep_1000/` (30 runs) | `eval_harness/evaluation/` | **Table 2** (estimator comparison, head+IS rows) |
| `gpt2_mixed_tail_proposal_sweep_250_debug/` (49 runs) | `eval_harness/evaluation/` | Table 2 mixed-tail proposal rows |
| `gpt2_is_sweeps_and_expanded_final/` (55 runs) | `omnilens/evaluation/` | Table 2 final head+IS/expanded evals (run dirs prefixed with their source sweep) |
| `gpt2_preemptable_sweep_1000/plots/pearson_kendall_token_level/{layerwise_pearson_kendall,rank_summary}.csv` | correlation scripts | `layerwise_pearson_kendall_heatmap` (appendix) |
| `gpt2_preemptable_sweep_1000/plots/top_token_agreement/{layerwise_top_token_agreement,rank_summary}.csv` | correlation scripts | `top_token_agreement_heatmap` (appendix) |
| `gpt2_preemptable_sweep_1000/plots/rank_ablation_exact_kl/rank_ablation_summary.csv` | eval aggregation | rank-ablation summary table/figure |
| `section4_figures/*.csv` (5 CSVs) | `plot_section4_subsetkl_figures.py` | Section-4 figures (objective x rank sweep, layerwise KL r64, head/tail budget frontier, heatmap manifest, training trajectories) |

**Tok/s and peak-GB cells of Table 2** come from the training logs — *reference
only, not copied*:
`omnilens/logs/gpt2_preemptable_sweep_1000/*.log` (48 logs) and
`omnilens/logs/gpt2_preemptable_is_sweep_1000/*.log` (24 logs)
(also `gpt2_preemptable_lowrank_{sweep,mc_sweep}_1000/` for the low-rank runs).

## 4. Upstream dependencies (not copied)

- **omnilens** (`/path/to/project/omnilens/`):
  the trainer package (`src/omnilens`) and the checkpoint tree
  (`src/checkpoints/`). The full-rank KL baseline referenced by the eval configs:
  raw trainer checkpoints now live at
  `src/checkpoints/gpt2/tuned/kl/residual/seed*/lens_step_1000.pt` (the tree was
  reorganized after the sweep; the historical path
  `src/checkpoints/gpt2_preemptable_sweep_1000/tuned_kl_baseline/lens_step_1000.pt`
  no longer exists). The translated tuned-lens-format baseline is at
  `eval_harness/my_lenses/gpt2_preemptable_sweep_1000/tuned_kl_baseline__lens_step_{250,500,750,1000}/`.
- **subset-kl** and **hookbox** packages on `PYTHONPATH` (training bootstraps them
  via `pip install git+https://github.com/pettyjohnjn/{subset-kl,hookbox}.git` in
  `jobs/lib/common.sh`; eval uses `/path/to/project/packages/subset-kl/src`).
- **The Pile** eval split via the HF cache
  (`HF_HOME=/path/to/hf_cache`).
- conda envs `omnilens` (train) and `tuned_lens_env` (eval/correlations) on the cluster.

## 5. Duplicate / stale notes (from the source inventory)

- `plot_gpt2_debug_search_metrics.py` exists both at the `eval_harness` root
  (stale duplicate, **not copied**) and in `run/tuned_lens/scripts/` (canonical,
  ships with the package).
- `eval.pbs` likewise exists at the `eval_harness` root and inside the
  package; both are here (`run/eval.pbs` and `run/tuned_lens/eval.pbs`) because
  the package was copied wholesale — `run/eval.pbs` is the one to submit.
- `eval_harness/evaluation/gpt2_debug_lowrank_sweep_1000/` exists but
  is **empty** (0 runs) — nothing to copy; `data/` has no subtree for it.
- `gpt2_is_sweeps_and_expanded_final/` re-evaluates runs that also appear in
  `gpt2_preemptable_lowrank_is_sweep_1000/` (final re-eval; run dirs carry a
  `<source_sweep>__` prefix). Prefer it for Table 2 final numbers.
- `omnilens/trash/` (old checkpoints/evals) was excluded, as were all
  `lens_step_*.pt`, `core.*` dumps (two ~388MB dumps sat in
  `tuned_lens/scripts/`), `__pycache__`, and PBS `.o/.e` logs.
- `run/jobs/submit/submit_gpt2_lowrank_sweep_part.sh` was added beyond the
  manifest list: `gpt2_lowrank_sweep_{1..4}.sh` are thin wrappers that `exec` it.

## 6. Seed study + RS baseline (added 2026-07-18)

15 new runs (all step 1000, single-GPU recipes identical to seed 0):
tuned+full-KL, LoRA-r64+full-KL, top-k k=1024, and Top-k+IS 512+512 at seeds
1–2, plus the RS baseline (`k_head=0` pure teacher sampling, RS-KD-style) at
tails 512/1024 x seeds 0–2 and 1536 x seed 0.

- Training: `run/gpt2_seed_study.pbs` + `run/gpt2_seed_worker.sh`; checkpoints
  in `omnilens/src/checkpoints/gpt2/.../seed{1,2}` and `subset_kl-is-k0-*`.
- Evaluation: `run/eval_gpt2_seed_study.pbs` -> `omnilens/evaluation/`
  `gpt2_seed_study_1000/` (identical Belrose loop, 131,072 Pile test tokens).
- Aggregation: `run/aggregate_seed_study.py` ->
  `data/evaluation/seed_study/seed_study_{per_run,summary}.csv`.
  Headline numbers: mean/early KL seed-sd <= 0.04 for every estimator; final KL
  sd 0.0007 (top-k) to 0.012 (IS; seed-0's 0.067 is the worst of 3); RS final
  KL 0.194 +- 0.001 (total 512), degrading to 0.66/1.07 at totals 1024/1536
  (unbiasedness of the k_head=0 estimator verified numerically — the
  degradation is an optimization effect).

**Throughput/peak-GB measurement.** `run/probe_rs_throughput.pbs` +
`run/probe_table2_throughput.pbs` measure every Table-2 config's Tok/s and
Peak GB with the released trainer (1 node, torchrun DDP x4, 50 steps); logs in
`omnilens/logs/probe_rs_*.log`. Table 2's Tok/s + Peak GB cells come from
these probes, so the table is reproducible with this code.
