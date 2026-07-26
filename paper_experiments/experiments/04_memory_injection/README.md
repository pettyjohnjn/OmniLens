# 04_memory_injection — Memory Injection (multi-hop) experiment family

Consolidated snapshot of the memory-injection experiments. Every file here was
copied (`cp -p`) from the canonical locations listed below; nothing upstream was
moved or modified. The paper plots in `../../plots/` (e.g.
`plot_lens_predicts_injection.py`, `plot_causal_lift.py`) read this `data/` copy
directly, so the folder regenerates its figures without the upstream trees.

Canonical source directory:
`/path/to/project/memory_injections/experiments/tweak_factor_analysis/`

---

## 1. What the experiment is

Multi-hop memory injection. Each example has an *implicit* prompt (requires a
latent fact) and an *explicit* prompt (states the fact). We form a **memory
vector** at a chosen layer — the explicit-minus-implicit difference of the
`resid_mid` activation at the final position — and **patch it back into the
implicit run** at scale `tau` (the "tweak factor"):

```
resid_mid[l]  <-  resid_mid[l] + tau * (h_explicit[l] - h_implicit[l])
```

Two roles are separated:

* **Lens = site selector (read task).** For each layer, a lens (logit lens,
  tuned lens, or low-rank lens) reads the patched `resid_mid` and reports
  P(answer). The lens's pick is `argmax_l [mean P_lens(ans | tweak=1) - mean
  P_lens(ans | tweak=0)]` — computed from two clean readout sweeps, so it is
  tau-independent.
* **Causal profile = ground truth (lens-independent).** With `--causal`, the
  runner patches each layer in turn, lets the model run to completion, and
  records the model's **own final P(answer)** per patched layer
  (`causal_P_ans_tau*`), plus **damage** metrics: `causal_KL_tau*` = KL(P_injected
  || P_clean) at the final position (nats), and `causal_top1keep_tau*` = fraction
  of examples whose argmax token is unchanged. Good site selection = high answer
  probability gained per nat of collateral distortion.

Runner outputs per (model, lens, dataset):
* `tweak_{N}.csv` — per-example lens readouts at tweak factor N (N = 0..10),
  columns `ans_prob_lens_edit_layer{l}` per layer + `answer_prob_obs`.
* `causal_profile.csv` — per-layer causal columns above (only with `--causal`).

Datasets: `hand` = 106 handwritten obscure-fact pairs
(`handwritten_obscure_explicit_data.csv`, the probe set of Sakarvadia et al.
2023, redistributed unmodified — see `LICENSE`); `2wmh` = 1000
2WikiMultiHop-style pairs (`multi_hop_1000.csv`).

## 2. How to run

Environment (the cluster):

```bash
module use /path/to/modulefiles; module load conda; conda activate omnilens
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/path/to/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

Canonical runner: `run/lens_tweak_factor_analysis.py` (pure HF + forward hooks;
no TransformerLens). Concrete example — LLaMA-3-8B, 2WMH, head+IS low-rank lens, with the
causal profile:

```bash
CK=/path/to/project/omnilens/src/checkpoints
python lens_tweak_factor_analysis.py \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --checkpoint $CK/models-meta-llama-meta-llama-3-8b-instruct/lora-r64/subset_kl-is-k512-tail1024/llama_expanded/init-default/seed0/lens_step_1000.pt \
  --lens-type lora --lora-rank 64 \
  --activation-site-preset llama_expanded --dtype bfloat16 \
  --dataset 2wmh --min-tweak 0 --max-tweak 10 --limit 200 \
  --causal --causal-batch 32 --causal-taus 1 2 4 \
  --output-dir results_llama8b_expanded/is_r64/2wmh
```

Other useful flags: `--lens-type {logit,tuned,lora}` (logit needs no
checkpoint), `--causal-only` (skip readout sweeps, just recompute the causal
profile), `--device-map auto` (shard 70B across 4 GPUs), `--omnilens-src`
(defaults to `/path/to/project/omnilens/src`).

**Path gotcha when running from this folder:** the runner resolves the dataset
directory as `<script>/../../data` (`REPO_ROOT/data`) and imports
`data.load_data`. In the canonical checkout that is
`/path/to/project/memory_injections/data/`. The copies in
`run/datasets/` are a reference snapshot; to execute from an arbitrary
location, recreate a `data/` dir two levels above the script (or just run the
canonical copy in place, as all PBS scripts do — they hard-code
`EXP=/path/to/project/memory_injections/experiments/tweak_factor_analysis`).

Analysis: `run/analyze_alllens_injection.py` scores every lens's site pick
against the causal profile (gain %, KL at pick, top-1 retention, efficiency =
P-gain per nat). Writes `alllens_injection_summary.csv`; with `--parity250` it
scores the 8B parity-250 lens picks instead and writes
`alllens_injection_summary_parity250.csv`.

### PBS scripts (all `#PBS -A YOUR_PBS_ACCOUNT`, mostly 1h debug-queue, 4x A100)

| Script | What it does / produces |
|---|---|
| `run_gpt2_alllens.pbs` | GPT-2: logit-lens sweeps (both datasets) + is_r64 causal re-runs with damage metrics; also runs GPT-2/8B DART jobs for the sibling toxicity experiment |
| `run_gpt2_causal_lenspred.pbs` | GPT-2 causal profiles + tuned/topk lens sweeps -> `results_gpt2_expanded/{tuned,topk_r64}` |
| `run_llama8b_alllens.pbs` | 8B: is_r64 causal re-runs (`--causal-only`, taus 1 2 4) + logit sweeps -> `results_llama8b_expanded/{is_r64,logit}` |
| `run_llama8b_causal_lenspred.pbs` | 8B causal profiles + topk lens sweeps |
| `run_llama8b_injection.pbs` | 8B head+IS/topk injection campaign (older `results_llama8b/` layout) |
| `run_llama70b_alllens_hand.pbs` / `run_llama70b_alllens_2wmh.pbs` | 70B (sharded, `--device-map auto`): is_r64 causal re-run with damage + logit sweep, per dataset -> `results_llama70b_expanded/` |
| `run_llama70b_causal_lenspred.pbs` / `run_llama70b_causal_2wmh.pbs` | 70B causal profiles (taus 1 2), per dataset |
| `run_llama70b_injection.pbs` / `run_llama70b_campaign.pbs` | 70B is_r64 injection sweeps (campaign = 2h prod-queue combined run incl. sibling DART) |
| `smoke_llama8b_injection.pbs` / `smoke_llama70b_injection.pbs` | Short smoke tests (tiny `--limit`) |
| `chain_alllens.sh` | Sequential qsub chain for the three alllens jobs |
| `run_llama8b_parity250_evalB.pbs` | **parity250 provenance**: 8B injection sweeps for the two 250-step parity low-rank lenses (head+IS, topk), one (lens, dataset) per GPU -> `results_llama8b_expanded/{is_parity250,topk_parity250}`. Feeds the 8B low-rank columns of the lens-prediction table |

The parity-250 setup: for the paper's tables all 8B trained lenses share an
**identical 250-step annealed schedule** (`seed0-parity250` checkpoints);
`run_llama8b_parity250_evalB.pbs` runs the injection sweeps with those lenses,
and `analyze_alllens_injection.py --parity250` scores them (the causal profile
is lens-independent and read from `is_r64`). `tuned_parity250/` comes from the
companion phase-A parity job in the same campaign
(`run_llama8b_parity250_evalA.pbs`, kept with the toxicity experiment it
primarily drives).

## 3. Data map

`data/` preserves the canonical layout `results_<model>_expanded/<lens>/<dataset>/*.csv`
(datasets: `hand`, `2wmh`; files: `tweak_0.csv`..`tweak_10.csv`, plus
`causal_profile.csv` under `is_r64`).

| Artifact | Consumed by |
|---|---|
| `results_*_expanded/is_r64/*/causal_profile.csv` + `tweak_*.csv` | **tab:causal_inject** + **fig_causal_lift** (causal P(answer) lift and damage per model) |
| 8B `{logit,is_parity250,topk_parity250,tuned_parity250}` + GPT-2/70B lens dirs + all `is_r64` causal profiles | **tab:lenspred** + **fig_site_selection** + **fig_lens_predicts_injection** (lens site picks scored against causal profiles) |
| `results_*_expanded/is_r64/*/tweak_*.csv` sweeps | **fig_injection_heatmaps** / **fig_injection_curves** (P(answer) vs layer x tweak factor) |
| `data/alllens_injection_summary_parity250.csv` | Output of `analyze_alllens_injection.py --parity250`; source table behind tab:lenspred numbers |
| `data/alllens_injection_summary.csv` | Same analysis with the original (non-parity) 8B lenses; kept for comparison |
| 8B `is_r64_step250/`, `tuned8b/` | Auxiliary earlier sweeps (step-250 head+IS before the parity re-train; first tuned-lens run); not referenced by the analyzer |
| GPT-2 `tuned/`, `topk_r64/`; 8B `topk_r64/`; 70B `logit/` | Additional lens variants entering the all-lens comparison via the analyzer's default (non-parity) mode |

Plot scripts live one level up (`../plot_lens_predicts_injection.py`,
`../make_section_figures.py`, ...) and write into `../../plots/`; they read the
canonical upstream results, not this snapshot.

## 4. Upstream dependencies NOT copied

* **Lens checkpoints** (`CK = /path/to/project/omnilens/src/checkpoints`):
  * GPT-2 head+IS: `$CK/gpt2/lora-r64/subset_kl-is-k512-tail1024/gpt2_expanded/init-default/seed0/lens_step_900.pt`
  * GPT-2 TopK: `$CK/gpt2/lora-r64/subset_kl-topk-k512/gpt2_expanded/init-default/seed0/lens_step_500.pt`
  * GPT-2 tuned: `$CK/gpt2/tuned/kl/gpt2_expanded/seed0/lens_step_500.pt`
  * 8B head+IS (main): `$CK/models-meta-llama-meta-llama-3-8b-instruct/lora-r64/subset_kl-is-k512-tail1024/llama_expanded/init-default/seed0/lens_step_1000.pt`
  * 8B TopK (original): `$CK/meta-llama-3-8b-instruct/lora-r64/subset_kl-topk-k512/llama_expanded/init-default/seed0/lens_step_250.pt`
  * 8B parity250 head+IS/TopK: `$CK/models-meta-llama-meta-llama-3-8b-instruct/lora-r64/subset_kl-{is-k512-tail1024,topk-k512}/llama_expanded/init-default/seed0-parity250/lens_step_250.pt`
  * 8B parity250 tuned: `$CK/models-meta-llama-meta-llama-3-8b-instruct/tuned/kl/residual/seed0-parity250/lens_step_250.pt` (used by evalA's GPU3 leg that produced `tuned_parity250/`)
  * 70B head+IS: `/path/to/project/lens_bundle/llama70b_lens/lens_step_250.pt`
* **Models**: 8B `meta-llama/Meta-Llama-3-8B-Instruct`;
  70B `meta-llama/Llama-3.3-70B-Instruct`;
  GPT-2 from the offline HF cache (`HF_HOME=/path/to/hf_cache`).
* **omnilens source** on `sys.path`: the runner imports `omnilens.lenses` /
  `omnilens.training.*` from `--omnilens-src`
  (default `/path/to/project/omnilens/src`).
* **Conda env** `omnilens` (torch + transformers; no transformer_lens needed).

## 5. Stale code (in the canonical dir; deliberately NOT copied)

* `injection_tweak_factor_analysis.py` and
  `injection_tweak_factor_analysis_original.py` — superseded TransformerLens-era
  runners; replaced by `lens_tweak_factor_analysis.py` (HF + hooks).
* `qsub_lens_tweak_factor.pbs` — references dead checkpoint paths.
* `qsub_tweak_factor_lens_debug.pbs` — args no longer match the current runner CLI.
* `experiment.pbs`, `experiment_parallel.pbs` — drivers for the old
  TransformerLens runner.
* Legacy result dumps `data/` and `multi_1000/` inside the canonical experiment
  dir — superseded TransformerLens outputs; the `results_*_expanded/` trees
  copied here replace them. (Older HF-era `results_llama8b/`, `results_llama70b/`
  and `results_*_smoke/` trees were likewise left upstream.)
* `results_*_expanded/archive_*` subdirs (pre-parity checkpoint runs:
  `archive_is_r64_step900` for GPT-2, `archive_is_r64_step1000` for 8B) —
  excluded from this snapshot.
