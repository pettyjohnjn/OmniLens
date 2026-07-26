# 05 — Toxicity: DART localization, ToxIn ablation, whole-model audit

Consolidated copy of the toxicity experiment family. Sources (read-only, copied
with `cp -p`, never modified):

- code + PBS + result CSVs: `/path/to/project/interp/experiments/toxicity_ablation/`
- GPT-2 900-step evaluation trees: `/path/to/project/evaluation_900step/`

Layout: `run/` (code + PBS), `data/` (result trees, relative structure preserved).

## 1. What this family is

Four linked studies on GPT-2, Llama-3-8B-Instruct, and Llama-3.3-70B-Instruct:

- **DART** (per-head toxic-token localization). For each attention head, project
  the last-token head output through a lens at the layer's `attn_out` site,
  take the top-50 tokens, and count hits against `run/toxic_dictionary.txt`,
  accumulated over toxic prompts (wiki_toxic, label==1). Output: a
  `[num_layers x num_heads]` counts CSV per lens. `run/llama_dart_localization.py`
  is the architecture-aware version (GQA-aware: an `o_proj` pre-hook splits into
  *query* heads, since GQA only shares K/V) and handles all three models;
  `run/find_toxic_heads_per_method.py` is the original GPT-2-native version
  (uses the `interp` library).

- **ToxIn** (targeted intervention, `run/llama_toxin_ablation.py`). Two
  interventions keyed off the DART counts: (a) ZERO-ABLATE the top-15 flagged
  heads, with a random-head control; (b) SOFT-SUBTRACT a toxic direction at the
  flagged heads' layers, swept over scale lambda. The direction is the
  normalized mean of the `lm_head` rows of toxic-vocabulary tokens (the
  *unembed* direction). Toxicity metric: Toxic-BERT on greedy continuations;
  cost budget: WikiText-2 perplexity.

- **Whole-model audit** ("allsites"/"allhooks", `run/llama_allsites_audit.py`
  for 8B; `run/find_toxic_sites_per_method.py` + `run/run_hookset_ablation.py`
  for GPT-2). Score all 6 x L sites (`attn_in, attn_out, resid_mid, mlp_in,
  mlp_out, resid_post`) through the lens (`score` -> `toxic_sites_is.csv`),
  then per site type subtract the lens-mapped toxic direction at the top 10%
  flagged sites, swept over lambda (`ablate` -> `<site_type>/summary.csv`).

- **Cross-lens agreement** (`run/analyze_dart_crosslens.py`). Reads all
  `toxic_heads_*.csv` per model and writes concentration stats
  (`dart_crosslens_summary*.csv`: total counts, top head, top-1/top-5 share,
  late/early ratio) and pairwise rank agreement (`dart_crosslens_pairs*.csv`).

Key findings this family supports:

- **Direction from the unembed, selection from the lens.** The unembed toxic
  direction beats trained-lens directions as the *ablation* direction (write
  task); trained lenses add value for head/site *selection* (read task).
- **8B is concentrated, 70B is diffuse.** At 8B, toxic mass concentrates
  (top head L23.H24; head+IS parity250 top-1 share ~12.4%, top-5 ~35.2%); at 70B
  the DART profile is diffuse and head-level ablation gains little.
- **Detection != intervention.** The sites that best *read out* toxicity are
  not the best places to intervene; at 8B, `mlp_out` is the best intervention
  site type in the audit.
- GPT-2's selector-matrix numbers are compared at a fixed cost budget: values
  are **interpolated to the PPL = 1.10x operating point** (see Section 3).

## 2. How to run

Environment (the cluster): `module use /path/to/modulefiles; module load conda;
conda activate omnilens`, then

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/path/to/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

Pipeline at 8B (DART -> ToxIn -> audit), abbreviated from
`run/run_llama8b_parity250_evalA.pbs` and `run/run_llama8b_allsites.pbs`:

```bash
MODEL=meta-llama/Meta-Llama-3-8B-Instruct
IS=$CKR/lora-r64/subset_kl-is-k512-tail1024/llama_expanded/init-default/seed0-parity250/lens_step_250.pt

# 1) DART: per-head toxic counts through the lens
python run/llama_dart_localization.py \
  --model-name "$MODEL" --checkpoint "$IS" --lens-type lora --lora-rank 64 \
  --activation-site-preset llama_expanded --dtype bfloat16 --n-prompts 200 \
  --out-csv results_llama8b/dart/toxic_heads_is_parity250.csv
# (tuned lens instead: --lens-type tuned --activation-site-preset residual --site-template "{l}")

# 2) ToxIn: ablate the lens-flagged heads under a PPL budget
python run/llama_toxin_ablation.py \
  --model-name "$MODEL" \
  --dart-csv results_llama8b/dart/toxic_heads_is_parity250.csv \
  --dtype bfloat16 --n-top 15 --n-prompts 100 --max-new-tokens 20 \
  --ppl-windows 32 --lambdas 0.0 0.05 0.1 0.2 0.3 \
  --out-csv results_llama8b/toxin_is_parity250/ablation_summary.csv

# 3) whole-model audit: score all 6xL sites, then ablate per site type
python run/llama_allsites_audit.py score  ...   # -> allhooks/toxic_sites_is.csv
python run/llama_allsites_audit.py ablate ...   # -> allhooks/<site_type>/summary.csv
# exact args: run/run_llama8b_allsites.pbs (Phase A on 1 GPU, Phase B fanned over 4)

# 4) cross-lens agreement over all toxic_heads_*.csv
python run/analyze_dart_crosslens.py            # -> dart_crosslens_{summary,pairs}.csv
```

70B runs identically with `--device-map auto` and reduced budgets
(`--n-prompts 150` DART / `40` ToxIn, `--ppl-windows 16`, lambdas `0 0.1 0.3`).

What each PBS did (all `-q debug`, 1 node / 4 GPUs, account YOUR_PBS_ACCOUNT):

| PBS | What it did |
|---|---|
| `run_llama8b_dart.pbs` | First 8B DART pass: head+IS (then-current step-1000 ckpt) + Top-k lenses, 150 prompts. |
| `run_llama8b_dart_matched.pbs` | Budget-matched DART: head+IS at steps 200 and 300 (bracketing Top-k's 250) for fair concentration comparison. |
| `run_llama8b_toxin.pbs` | First 8B ToxIn on head+IS-selected heads, unembed direction, top-15, lambdas 0-0.3 -> `toxin/`. |
| `run_llama8b_toxin_selectors.pbs` | 8B head-*selector* comparison: same ToxIn on tuned-8B / logit / topk-selected heads (3 GPUs in parallel) -> `toxin_{tuned8b,logit,topk}/`. |
| `run_llama8b_allsites.pbs` | Whole-model audit: Phase A scores all 6x32 sites through the head+IS lens; Phase B per-site-type ablation sweeps -> `allhooks/`. |
| `run_llama8b_is300.pbs` | Standardizes 8B to head+IS@300: ToxIn + audit (+ injection, out of scope here) at `lens_step_300`; refreshed `toxic_heads_is_r64.csv`, `toxin/`, `allhooks/`. |
| `run_llama8b_parity250_tuned.pbs` | head+IS@250 (resume ckpt) + full-rank tuned-8B reference reruns: DART/ToxIn/audit-score -> `*_step250` outputs, `toxic_heads_tuned8b.csv`. |
| `run_llama8b_parity250_evalA.pbs` | **The parity250 refresh.** All three 8B lenses retrained on one shared 250-step cosine schedule (identical to 70B's); this job reran DART + ToxIn per lens onto NEW `*_parity250` paths — these are the CSVs the paper tables use. (Phase B, `run_llama8b_parity250_evalB.pbs`, covered injection sweeps — other family, not copied.) |
| `run_llama70b_dart.pbs` | 70B DART: head+IS lens + logit reference, 150 prompts, `device_map=auto`. |
| `run_llama70b_toxin.pbs` | 70B ToxIn on head+IS-selected heads (40 prompts, 16 PPL windows, lambdas 0/0.1/0.3). |
| `run_gpt2_is500_rerun.pbs` | GPT-2 standardized to a common 500-step budget: DART at head+IS@500 (+ injection, out of scope). |
| `submit_tox_matrix.pbs` | GPT-2 selector x direction ablation MATRIX with the 900-step low-rank lens (interp library path) -> `evaluation_900step/toxicity_ablation_matrix/`. |
| `submit_tox_heads_tuned.pbs` | Added the tuned head-selector ROW to the matrix (`heads_tuned/`, assembled from `_heads_tuned_tmp/`). |
| `submit_tox_tuned.pbs` | Filled the tuned ablation-direction COLUMN of the matrix. |
| `submit_tox_allhooks.pbs` | GPT-2 all-hooks audit with the 900-step lens -> `evaluation_900step/toxicity_allhooks_ablation/`. |

## 3. Data map — which file feeds which paper table/figure

| Paper element | Data here |
|---|---|
| `tab:dart`, `fig_dart_*` | `data/results_llama8b/dart/toxic_heads_{is,topk,tuned}_parity250.csv` (+ `toxic_heads_logit.csv`; GPT-2 and 70B `dart/` CSVs for the scale comparison) |
| `tab:selector` | `data/results_llama8b/toxin_{is,topk,tuned}_parity250/ablation_summary.csv` (+ `toxin_logit/` for the logit row) |
| `tab:ablation` | `data/results_llama8b/toxin_is_parity250/ablation_summary.csv` + `data/results_llama70b/toxin/ablation_summary.csv` |
| `tab:cross_lens` | `data/dart_crosslens_summary_parity250.csv`, `data/dart_crosslens_pairs_parity250.csv` |
| `fig_allsites` + audit prose | `data/results_llama8b/allhooks/` (and `allhooks_step250/`) + `data/evaluation_900step/toxicity_allhooks_ablation/` |
| GPT-2 selector numbers | `data/evaluation_900step/toxicity_ablation_matrix/` — **NOTE: these are interpolated to the PPL = 1.10x operating point**, not read off a single lambda row |

## 4. Upstream dependencies NOT copied here

- **Models**: GPT-2 = HF `gpt2`; 8B =
  `meta-llama/Meta-Llama-3-8B-Instruct`;
  70B = `meta-llama/Llama-3.3-70B-Instruct`.
- **Lens checkpoints** (under
  `CKR=/path/to/project/omnilens/src/checkpoints/models-meta-llama-meta-llama-3-8b-instruct`):
  - parity250 trio: `lora-r64/subset_kl-is-k512-tail1024/llama_expanded/init-default/seed0-parity250/lens_step_250.pt`,
    `lora-r64/subset_kl-topk-k512/llama_expanded/init-default/seed0-parity250/lens_step_250.pt`,
    `tuned/kl/residual/seed0-parity250/lens_step_250.pt`
  - audit lens (head+IS@300): `.../seed0/lens_step_300.pt`; head+IS@250-resume: `.../seed0-resume250/lens_step_250.pt`;
    tuned-8B reference: `tuned/kl/residual/seed0/lens_step_500.pt`
  - 70B lens bundle: `/path/to/project/lens_bundle/llama70b_lens/lens_step_250.pt`
  - GPT-2 head+IS@500: `.../checkpoints/gpt2/lora-r64/subset_kl-is-k512-tail1024/gpt2_expanded/init-default/seed0/lens_step_500.pt`;
    the 900-step GPT-2 low-rank lens is resolved via `interp.config.CKPT_LORA_R64`.
- **HF assets** (offline cache at `$HF_HOME` above): `OxAISH-AL-LLM/wiki_toxic`
  (toxic prompts), `wikitext` / `wikitext-2-raw-v1` (PPL), `unitary/toxic-bert`
  (toxicity classifier).
- **Libraries**: `omnilens` (`/path/to/project/omnilens/src`)
  for lens loading in the llama_* scripts; the GPT-2-native path
  (`find_toxic_*`, `run_hookset_ablation.py`, `toxicity_ablation.py`)
  additionally needs `interp` (`/path/to/project/interp/src`)
  and `/path/to/project/packages/subset-kl/src` on `PYTHONPATH`
  (note: `interp` needs `transformer_lens` in the env).
- **Deliberately not copied** (out of scope): `train_lora_zeroinit_900*.pbs`
  (lens *training*, not an experiment); `plot_final_figures.py` and
  `plot_llama_case_studies.py` (plot scripts live in `../../plots/`);
  PBS `.o`/`.e` logs; `results_figures/` and `plots_final/` (figure outputs);
  all `archive*` result dirs; `run_llama8b_parity250_evalB.pbs` (injection
  family); per-cell `toxicity_generations.csv` dumps and `_heads_tuned_tmp/`
  `result.json`s in the matrix tree (final assembled rows live in `heads_*/`).

## 5. Vintage notes (which results used which lens checkpoint)

- **Parity250 (current, used by the paper tables)**:
  `results_llama8b/dart/toxic_heads_{is,topk,tuned}_parity250.csv`,
  `results_llama8b/toxin_{is,topk,tuned}_parity250/`,
  `dart_crosslens_{summary,pairs}_parity250.csv`.
- **Pre-parity 8B** (mixed budgets; kept for provenance/comparison):
  `toxic_heads_is_r64.csv` (head+IS@300 after the is300 pass),
  `toxic_heads_is_r64_step250.csv` (head+IS@250 resume ckpt),
  `toxic_heads_topk_r64.csv` (Top-k@250 old schedule),
  `toxic_heads_tuned8b.csv` (tuned@500), `toxic_heads_logit.csv` (no training);
  `toxin/` (head+IS@300 heads), `toxin_step250/`, `toxin_logit/`, `toxin_topk/`,
  `toxin_tuned8b/`; `dart_crosslens_{summary,pairs}.csv` (pre-parity mix).
- **Audit is NOT parity250**: `results_llama8b/allhooks/` is still the
  **step-300 head+IS lens**; `allhooks_step250/` used the head+IS@250 *resume* checkpoint
  (not the parity250 retrain).
- **70B**: single vintage — the 250-step `lens_bundle` lens (the schedule the
  8B parity250 retrain copied).
- **GPT-2**: `results_gpt2/dart/` at the standardized 500-step budget
  (`toxic_heads_is_r64.csv` = head+IS@500 from the is500 pass);
  `data/evaluation_900step/` trees used the **900-step** GPT-2 low-rank lens.
- Step-1000 relics live only in the source's `archive*` dirs (not copied).

## 5. 8B seed stability (added 2026-07-18)

The parity250 8B lenses (head+IS 512+1024, topk k512, tuned full-KL) were retrained
at seeds 1–2 under the identical 250-step schedule and re-audited:

- Job: `run/run_llama8b_seed_stability.pbs` — DART (200 prompts) per lens x
  seed -> `results_llama8b/dart/toxic_heads_{is,topk,tuned}_parity250_s{1,2}.csv`,
  plus injection readouts (tweak 0/1, 2WMH, 200 ex) ->
  `tweak_factor_analysis/results_llama8b_expanded/{lens}_parity250_s{1,2}/2wmh/`
  (seed 0 = the existing parity250 outputs).
- Aggregation: `run/analyze_8b_seed_stability.py` ->
  `data/seed_stability/seed_stability_{dart,injection}.csv`.
- Headlines: L23.H24 is the top DART head in 9/9 lens x seed audits; top-5
  overlap 4–5/5 across seeds (tuned rankings near-deterministic, Spearman 0.98);
  head+IS top-5 share 31–35% vs topk 11–20%. Injection Delta_l pick: head+IS {19,18,18}
  (interior, 90–100% captured at tau=2), topk {31,10,31}, tuned {22,31,24} —
  only the Top-k+IS selection is seed-robust.
