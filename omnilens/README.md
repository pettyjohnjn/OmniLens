# OmniLens

**Scalable lens-based interpretability for large language models.**

OmniLens trains *lenses* — auxiliary decoders that map an intermediate
activation to a distribution over the model's vocabulary — densely across a
model, rather than at a few preselected components. It combines two
ingredients:

- **Low-rank translators.** One translator architecture, applied at every
  supported model-width hookpoint (residual stream, attention output, MLP
  output), with `O(rd)` trainable parameters per hookpoint instead of `O(d^2)`.
- **Subset-KL.** A family of training objectives that computes the loss from a
  selected subset of vocabulary items rather than materializing the full lens
  distribution, which is what makes training tractable at large vocabulary
  sizes.

## Architecture

```
omnilens/
├── hooks/          # Re-exports from hookbox (activation capture)
├── losses/         # Adapters over subset-kl + local losses
├── lenses/         # Lens modules (logit, tuned, low-rank)
├── ops/            # Wrapper for indexed_logits CUDA extension
├── training/       # Orchestration layer (trainer, FSDP, streaming loader)
├── data/           # Data loading and chunking
└── cli/            # Command-line interface
```

### Sub-packages

Three pieces of infrastructure live in standalone packages, vendored under
`../packages/` in this repository so no network access is needed:

| Package | Role |
|---------|------|
| **hookbox** | Activation capture, hook management, distributed model unwrapping. `omnilens.hooks` is a thin re-export shim. |
| **subset-kl** | Core KL math: `SubsetKLLoss`, `KLDivergenceLoss`, head/tail estimators, and functional APIs (`select_topk_indices`, `subset_kl_from_gathered`). `omnilens.losses` wraps these with adapters that add a `labels` parameter for cross-entropy compatibility. |
| **indexed_logits** | Optional CUDA kernel computing `out[i,j] = dot(H[i,:], W[idx[i,j],:])` without materializing `[N, k, d]` tensors. `omnilens.ops` wraps it with a `torch.gather` fallback. |

`omnilens` itself provides the lenses (`LogitLens`, `TunedLens`,
`LowRankLens`), `SharedSubsetKLLoss`, `CrossEntropyLoss`, training
orchestration, data loading, and the CLI.

## Installation

```bash
pip install ../packages/hookbox
pip install ../packages/subset-kl
pip install ../packages/indexed_logits   # optional, needs a CUDA toolchain
pip install -e .                         # omnilens itself
pip install -e '.[dev]'                  # + pytest
```

`indexed_logits` is optional: without it, `omnilens.ops` falls back to
`torch.gather` and everything still runs, just more slowly.

## Quick start

```python
from omnilens.hooks import ActivationCollector
from omnilens.losses import create_loss
from omnilens.lenses import create_lens
from omnilens.training import LensTrainer, TrainConfig

collector = ActivationCollector(model)
loss_fn = create_loss("subset_kl", k=256)
lens = create_lens("lora", layer_ids=range(12), hidden_size=768, unembed=unembed, r=16)

trainer = LensTrainer(model, lens, loss_fn, collector, config, ddp_state, amp_ctx)
trainer.train(dataloader_factory, optimizer)
```

From the command line:

```bash
omnilens train --model_name gpt2 --lens_type lora --loss_type subset_kl \
               --subset_kl_mode is --subset_kl_k 512 --subset_kl_k_tail 512
```

## Losses

| Loss | Memory | Use case |
|------|--------|----------|
| `kl` | O(B·T·V) | Full-KL baseline, small models |
| `subset_kl` | O(B·T·k) | Recommended default |
| `shared_subset_kl` | O(B·chunk·K) | Largest models |
| `ce` | O(B·T·V) | Label-based training |

### Subset-KL estimators (`--subset_kl_mode`)

| Mode | What it computes |
|------|------------------|
| `topk` | Renormalized KL over the top-k teacher tokens. Biased, cheapest, recommended default. |
| `is` | Exact KL on the top-k head plus an importance-sampled estimate of the tail. Unbiased gradients. |
| `k2` | Top-k head KL plus a Schulman K2 squared-error tail penalty. |
| `k3` | Top-k head KL plus a Schulman K3 tail estimator. |

Set `--subset_kl_k_tail > 0` to enable the tail term for `is`/`k2`/`k3`.

> **Naming note.** `is` is the head+IS estimator — "Top-k+IS" in the paper:
> exact evaluation on the top-k head plus an importance-sampled tail estimate.
> Run directories and result labels under `../paper_experiments/` use the same
> `is` / `head+IS` naming.

## Layout

```
src/omnilens/    the package
jobs/            job launcher library, per-run .env files, PBS submit scripts
scripts/         benchmark, analysis, and training scripts
tests/           pytest suite
benchmarks/      memory/throughput summaries produced by scripts/bench_*.pbs
```

Job scripts target a PBS Pro cluster and contain placeholder paths
(`/path/to/...`) and a placeholder account (`YOUR_PBS_ACCOUNT`); they are
included as a record of how the reported runs were configured rather than as
turnkey scripts.

## Tests

```bash
pytest tests/
```
