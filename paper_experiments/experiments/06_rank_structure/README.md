# 06_rank_structure — Is the learned translator low rank?

Evidence for the paper's low-rank premise (Section 3, `fig_rank_structure`). The
question is whether a rank-$r$ translator works because the full-rank solution is
itself compressible, or for some other reason. The answer turns out to be the
latter, and the distinction matters: it is why the paper describes low rank as a
constraint on training rather than a compression of a dense solution.

## 1. Two measurements, two different questions

**Spectral (the matrix view).** SVD of the deviation $\Delta$ learned by a
full-rank tuned lens — `translators.<layer>.weight`, since the forward is
`h + translator(h)`, so the stored weight *is* the deviation. Sourced from
`omnilens/analysis/intrinsic_dimensionality/` (May 2026) and vendored
here as `data/*_spectrum_layer_summary.csv`. It says $\Delta$ is **not** low rank:
rank 64 holds 58.5% of its Frobenius energy at GPT-2 and 35.4% at 8B, numerical
rank stays near $d$ (756/768 and 3989/4096), and both get *worse* over training.

**Functional (the behavior view).** `run/eval_rank_truncation.py` asks the
question the objective actually cares about — how much KL does a rank-$r$ lens
lose — with no linearization and no choice of matrix norm. Three variants scored
against one teacher pass:

| variant | what it is |
|---|---|
| `trained_r{r}` | a low-rank lens trained at rank `r` |
| `truncated_r{r}` | the full-rank lens with $\Delta$ cut to rank `r` by SVD |
| `random_r{r}` | the same $\Delta$ projected onto a random `r`-dim subspace |

The random control is what makes a positive result non-vacuous: it separates
"the top-$r$ singular subspace is special" from "any $r$ directions would do."

## 2. Result

At GPT-2 rank 64, against a full-rank final-layer KL of 0.0750:

- trained **0.0796** (+6.2%)
- truncated **0.1151** (+53.5%)
- random **1.1909** (15.9x)

Trained beats truncated at all nine ranks. Replicated on three corpora
(shuffled Pile-val, Pile file-order, WikiText-2); all agree on ordering and on
ratios within a few percent. `data/meta*.json` records each protocol.

## 3. How to run

```bash
# data (one debug-queue job, ~5 min on one A100)
cd run && qsub eval_rank_truncation.pbs

# figure
cd ../../../plots && python plot_rank_structure.py
```

The eval needs lens checkpoints that are **not** copied here:
`omnilens/src/checkpoints/gpt2/tuned/kl/residual/seed0` (full-rank)
and `gpt2/lora-r{1,4,...,384}/kl/residual/init-default/seed0` (the trained
sweep), both at step 1000. Translators are stored bf16 and cast to fp32 before
SVD — bf16 carries about three decimal digits and would corrupt the small
singular values the tail behavior depends on.

`data/llama8b_spectrum_layer_summary.csv` is the 8B spectral data. No 8B
functional replication yet; the matched pair for it would be
`meta-llama-3-8b-instruct/tuned/kl/residual/seed0` and
`lora-r64/kl/residual/init-default/seed0`, which share step 160.
