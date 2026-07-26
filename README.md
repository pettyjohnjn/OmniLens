# OmniLens — code and experiment artifacts

Anonymized supplementary code for the paper *OmniLens: dense trained-lens
coverage for large language models*.

A **lens** decodes an intermediate activation into a distribution over the
model's vocabulary. OmniLens makes it affordable to train lenses *densely* —
at every supported residual, attention, and MLP hookpoint — rather than at a
few preselected components, by combining low-rank translators with the
Subset-KL family of training objectives.

## What is in this repository

```
omnilens/           The training framework: low-rank + full-rank lenses,
                    Subset-KL objectives, FSDP/streaming trainer, CLI, tests.
                    See omnilens/README.md.

packages/           Three standalone sub-packages the framework depends on,
                    vendored here so the artifact is self-contained:
  hookbox/            activation capture and hook management
  subset-kl/          the core subset-KL math
  indexed_logits/     optional CUDA kernel for gathered logit computation

paper_experiments/  Everything behind the figures: one directory per
                    experiment family with its run scripts and the result
                    CSV/JSON the plots read, plus every plot script and the
                    shared Matplotlib style.
                    See paper_experiments/README.md.
```

## Reproducing

Install the framework and its sub-packages (see `omnilens/README.md` for
detail):

```bash
cd omnilens
pip install ../packages/hookbox ../packages/subset-kl
pip install ../packages/indexed_logits    # optional, needs a CUDA toolchain
pip install -e '.[dev]'
pytest tests/
```

Train a dense low-rank lens stack on GPT-2 with the Top-k+IS objective:

```bash
omnilens train --model_name gpt2 --lens_type lora --lora_rank 64 \
               --loss_type subset_kl --subset_kl_mode is \
               --subset_kl_k 512 --subset_kl_k_tail 512 \
               --activation_site_preset residual
```

Regenerate every data-driven figure — all except the static diagram
`fig_method_overview` — from the shipped result data (no GPU, no model
downloads; the plot scripts read only the CSV/JSON under
`paper_experiments/experiments/*/data`):

```bash
cd paper_experiments/plots
./run_all.sh
```

`paper_experiments/README.md` maps each figure in the paper to the plot script
and experiment directory that produce it.

## Notes for reviewers

**Naming.** `--subset_kl_mode is` selects the head+IS estimator ("Top-k+IS" in
the paper): the KL is evaluated exactly on the top-k head and estimated on the
remaining vocabulary with importance-sampled draws. Run directories and result
labels carry `is` / `head+IS` accordingly (for example
`gpt2_preemptable_is_sweep_1000__lora_subset_is_r64_k512_tail512`), and the run
scripts, shipped data, and plot scripts all use these names consistently.

`lora` is the identifier for the low-rank lens type (`--lens_type lora`, class
`LowRankLens`), since the translator is a LoRA-style low-rank update.

**Cluster scripts.** The `jobs/` and `run/` directories record how the reported
runs were configured on a PBS Pro cluster. They contain placeholder paths
(`/path/to/...`), a placeholder allocation (`YOUR_PBS_ACCOUNT`), and
placeholder model locations; they document the experimental configuration
rather than run turnkey. The Python entry points they invoke are the real
interface.

**Scale.** The 405B run reported in the paper uses the streaming FSDP loader in
`omnilens/src/omnilens/training/stream_load.py`; its configuration is recorded
in `omnilens/jobs/env/llama3_405b_*.env`.

**Third-party code.** `paper_experiments/experiments/01_rank_ablation_estimator/run/tuned_lens/`
is a modified copy of the `tuned-lens` package (Belrose et al., 2023), included
so that the full-rank tuned-lens baseline is scored by the original evaluation
loop rather than a reimplementation. It remains under its own copyright. The
datasets under `paper_experiments/experiments/04_memory_injection/run/datasets/`
are likewise redistributed under their original licenses: the multi-hop set is
derived from 2WikiMultiHop, and the handwritten obscure/explicit probe set is
the one introduced by Sakarvadia et al. (2023), reused unmodified so the
memory-injection case study runs on exactly the original prompts. See `LICENSE`.

## License

MIT — see `LICENSE`, which also records the third-party components above.
