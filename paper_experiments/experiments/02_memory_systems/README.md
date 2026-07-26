# 02_memory_systems — Memory / Systems Benchmarks

Consolidated experiment folder for the MEMORY/SYSTEMS BENCHMARKS family of the
OmniLens paper. Sources copied read-only (`cp -p`) from
`/path/to/project/omnilens/`.

## 1. What this family measures

The paper's feasibility argument rests on a **two-lever memory model** with two
independent buckets:

- **Persistent optimizer bucket** — parameters + gradients + Adam state for the
  trainable lens, scaling as L·d² per site. Training full translator matrices
  at 70B scale hits a hard wall here; **LoRA (r=64) removes the wall** by
  shrinking the trainable state by orders of magnitude. This bucket is exact
  parameter arithmetic — no measurement needed (see fig_memory_optimizer).
- **Transient readout bucket** — the loss-head activations, scaling with
  microbatch × context length × vocab (B·T·V for full KL). This bucket grows
  with sequence length and dominates at large context; **Subset-KL (top-k /
  top-k+IS) extends the ceiling** by replacing the V-sized readout with a
  k-sized one.

Which lever dominates flips between small and large models (GPT-2 → 70B), so
both are needed (Appendix Memory_Model, tab:needboth).

**Tier-1 (measured, 8B single-GPU):** `run/bench_readout_8b.pbs` runs 12
configs — {fullkl, topk, is} × seq {1024, 2048, 4096, 8192} — each on a single
A100-40GB (teacher + lens + readout on one GPU), fixed microbatch
(batch_size 2), 3 optimizer steps, on the `llama_expanded` hookset of
Llama-3-8B-Instruct. Per-GPU peak memory and throughput are recorded per
config; **OOMs are recorded as results** (STATUS=OOM), since the OOM frontier
is exactly what the readout-bucket model predicts. Result: full-KL OOMs at
seq 4096 while top-k still fits at 34.4 GB, and top-k+IS at 4096 records a
38.8 GB peak before OOM — the readout lever in action.

**Tier-2 (measured, 70B multi-node):** `run/bench_70b_fullkl.pbs` runs
LoRA r64 + **full-KL** on the 70B `llama_expanded` hookset across 24 nodes
(FSDP-style multinode model parallel, 96 GPUs), with a microbatch identical to
the production run (batch_size 2, seq 1024), 2 optimizer steps — just enough
to record true per-rank peak memory. Measured peak: **35.47 GB** (metrics.csv
step 2), vs the production LoRA + Subset-KL (top-k+IS, k=512, tail 1024) run's
**34.70 GB** — i.e. full-KL at 70B fits on a knife-edge on 40 GB parts at this
microbatch, quantifying the full-KL feasibility cell (tab:reconcile).

All peak numbers are `torch.cuda.max_memory_allocated` reported in **decimal
GB**, taken on the metrics-writing rank.

The earlier feasibility probes (`probe_llama3_70b_feasibility_memory.py`,
`summarize_llama3_70b_feasibility_direct.py`,
`summarize_llama3_70b_feasibility_proxy.py`, `run_with_rss_sampler.py`) are the
May-2026 precursors to the tiered benchmarks: host-RSS-sampled loading probes
and their direct/proxy summarizers.

**Measurement harness note (not copied here):** the benchmark harness is the
omnilens trainer itself — `src/omnilens/cli/main.py` +
`src/omnilens/training/trainer.py` in the upstream `omnilens` repo,
whose trainer writes `peak_memory_gb` / `tok_per_sec` into `metrics.csv` every
`log_every` steps. The PBS scripts here only orchestrate it.

## 2. How to run

Do **not** rerun from this folder — paths inside the PBS scripts point at the
upstream repo. To reproduce, submit from `omnilens/scripts/`:

- **Tier-1:** `qsub bench_readout_8b.pbs` — 1 node × 4 GPUs (`debug` queue,
  1 h walltime, account YOUR_PBS_ACCOUNT). Runs the 12 configs in 3 rounds of 4 parallel
  single-GPU jobs. Env: conda `omnilens`, `~/.hf_env`,
  `HF_HOME=/path/to/hf_cache`, `PYTHONPATH=$REPO/src`. Each
  config writes `<loss>_s<seq>/{metrics.csv, run_config.csv,
  activation_sites.csv, run.log, STATUS}`; an inline-heredoc Python step then
  aggregates last-step `peak_memory_gb` / `tok_per_sec` + STATUS into
  `summary.csv`.
- **Tier-2:** `qsub bench_70b_fullkl.pbs` — 24 nodes × 4 GPUs (`prod` queue,
  1 h walltime). torch.distributed.run rendezvous over the PBS nodefile;
  writes `benchmarks/readout_70b_fullkl/{metrics.csv, run_config.csv,
  activation_sites.csv}` (plus a lens checkpoint, not copied here).

## 3. Data map

| Data | Feeds |
|---|---|
| `data/benchmarks/readout_scaling_8b/summary.csv` + per-config `metrics.csv` | `fig_memory_readout` (measured 8B peak vs context per objective) and the Appendix Memory_Model tables (`tab:needboth`, `tab:reconcile`) |
| `data/benchmarks/readout_70b_fullkl/metrics.csv` | the 70B full-KL **35.5 GB** cell of `tab:reconcile` |
| `data/llama70b_production_run/metrics.csv` | the **34.7 GB** production LoRA+Subset-KL claims (peak_memory_gb = 34.6998 across all 250 steps) |
| (no data) `fig_memory_optimizer` | optimizer-bucket bars are exact parameter arithmetic from model/lens shapes — no measurement needed |

Per-config layout under `data/benchmarks/readout_scaling_8b/<loss>_s<seq>/`:
`metrics.csv` (per-step peak/throughput), `run_config.csv` (full CLI config),
`activation_sites.csv` (hookset), `STATUS` (OK/OOM), `run.log` (trainer log).
Lens checkpoints (`lens_step_*.pt`, ~205 MB each at 8B, ~1 GB each at 70B)
were deliberately **not** copied.

## 4. Upstream dependencies

- **Trainer/harness:** `omnilens` repo (`omnilens.cli.main train`),
  conda env `omnilens`.
- **Models:**
  8B — `meta-llama/Meta-Llama-3-8B-Instruct`;
  70B — `meta-llama/Llama-3.3-70B-Instruct`.
- **Data:** Pile (`--data_source pile`), streamed via
  `HF_HOME=/path/to/hf_cache`.

## 5. Caveats (as disclosed in the paper)

- Peaks are `torch.cuda.max_memory_allocated` (**allocated**, not reserved);
  the allocator's reserved pool is higher, so real headroom on a 40 GB part is
  tighter than the allocated numbers suggest. The 35.5 vs 34.7 comparison is
  apples-to-apples within this convention.
- 70B numbers are measured on the **metrics-writing rank (rank 0)** only;
  per-rank peaks can differ slightly under multinode sharding.
- GB are decimal (10⁹ bytes).
- Tier-1 is at fixed microbatch (batch_size 2); the readout bucket scales
  linearly in microbatch, so OOM frontiers shift with B.
- `is_s4096` has STATUS=OOM yet a recorded 38.78 GB metrics row: it survived
  early steps and OOM'd later — the recorded peak is a lower bound on demand.
