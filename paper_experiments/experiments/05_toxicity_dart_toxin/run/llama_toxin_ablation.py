"""ToxIn ablation for LLaMA (8B / 70B): does suppressing DART-flagged heads reduce
toxicity, and does a random-head control not?

Two interventions, both keyed off the DART toxic-head counts we already produced:
  * ZERO-ABLATE  the top-N flagged heads (hard); random-N control for comparison.
  * SOFT-SUBTRACT a toxic direction at the flagged heads' layers, swept over scale
    lambda (the ToxIn curve).

Per our finding that the *unembed* is the best ablation *direction* while the lens
is best for head *selection*, the toxic direction is the (normalised) mean of the
lm_head rows of toxic-vocabulary tokens, and head selection comes from DART.

Toxicity = Toxic-BERT on greedy generations; perplexity = WikiText-2. Model-agnostic
(GPT-2/LLaMA), single-GPU or device_map=auto (70B). Cut for speed, not rigour.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch


def load_dart_heads(csv_path: Path, n_top: int, n_layers: int, n_heads: int, seed: int):
    d = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            l = int(row["layer"])
            for h in range(n_heads):
                d[(l, h)] = float(row[f"head_{h}"])
    ranked = sorted(d, key=lambda k: d[k], reverse=True)
    top = ranked[:n_top]
    rng = np.random.default_rng(seed)
    # random control drawn from the non-top heads
    pool = [k for k in d if k not in set(top)]
    rand = [tuple(pool[i]) for i in rng.choice(len(pool), size=n_top, replace=False)]
    return top, rand


def toxic_direction(model, tokenizer, toxic_ids, device):
    W = model.lm_head.weight.detach().float()          # [V, d]
    ids = torch.tensor(sorted(toxic_ids), device=W.device)
    d = W[ids].mean(0)                                  # [d]
    d = d / (d.norm() + 1e-8)
    return d.to(device)


class HeadIntervention:
    """Pre-hook on each flagged layer's o_proj input. mode='zero' zeroes the flagged
    heads; mode='subtract' leaves heads intact (subtraction done at attn_out post-hook)."""

    def __init__(self, model, heads, num_heads, head_dim):
        self.bylayer = {}
        for (l, h) in heads:
            self.bylayer.setdefault(l, []).append(h)
        self.num_heads, self.head_dim = num_heads, head_dim
        self._model = model
        self._handles = []

    def _zero_hook(self, hs):
        def hook(_m, inputs):
            x = inputs[0].clone()
            b, s, _ = x.shape
            xh = x.view(b, s, self.num_heads, self.head_dim)
            for h in hs:
                xh[:, :, h, :] = 0.0
            return (xh.view(b, s, -1),) + inputs[1:]
        return hook

    def __enter__(self):
        for l, hs in self.bylayer.items():
            mod = self._model.model.layers[l].self_attn.o_proj
            self._handles.append(mod.register_forward_pre_hook(self._zero_hook(hs)))
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles = []


class SoftSubtract:
    """Post-hook on flagged layers' o_proj output: subtract lambda*||h||*d_tox from
    the attention output (norm-relative), the ToxIn soft intervention."""

    def __init__(self, model, layers, d_tox, scale):
        self.layers = sorted(set(layers))
        self.d_tox = d_tox
        self.scale = scale
        self._model = model
        self._handles = []

    def _hook(self):
        d = self.d_tox
        s = self.scale
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            nrm = h.norm(dim=-1, keepdim=True)
            h = h - s * nrm * d.to(h.dtype).to(h.device)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook

    def __enter__(self):
        for l in self.layers:
            mod = self._model.model.layers[l].self_attn.o_proj
            self._handles.append(mod.register_forward_hook(self._hook()))
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.no_grad()
def measure_toxicity(model, tokenizer, classifier, prompts, device, max_new_tokens):
    scores, ntox = [], 0
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=True, max_length=200).input_ids.to(device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        r = classifier(gen[:512] or ".", truncation=True)[0]
        is_tox = r["label"].lower() == "toxic"
        scores.append(r["score"] if is_tox else 1.0 - r["score"])
        ntox += int(is_tox)
    return sum(scores) / len(scores), ntox / len(prompts)


@torch.no_grad()
def measure_perplexity(model, tokenizer, device, n_windows, max_len):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    nlls = []
    for begin in range(0, len(ids) - max_len, max_len):
        chunk = ids[begin:begin + max_len].unsqueeze(0).to(device)
        nlls.append(model(chunk, labels=chunk).loss.item())
        if len(nlls) >= n_windows:
            break
    return math.exp(sum(nlls) / len(nlls))


def _load_toxic(n_prompts):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tox_common import load_toxic_words, load_toxic_prompts, TOXIC_DICT
    return load_toxic_words(TOXIC_DICT), load_toxic_prompts(n_prompts)


def _toxic_ids(tokenizer, toxic_words):
    V = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    return {t for t in range(V) if tokenizer.decode([t]).strip().lower() in toxic_words}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--dart-csv", required=True, help="DART toxic-head counts CSV.")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--n-top", type=int, default=15)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--ppl-windows", type=int, default=32)
    ap.add_argument("--ppl-max-len", type=int, default=1024)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.device_map:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, low_cpu_mem_usage=True, torch_dtype=dt, device_map=args.device_map).eval()
        device = model.lm_head.weight.device
    else:
        device = torch.device(args.device)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, low_cpu_mem_usage=True, torch_dtype=dt).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cls_dev = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model="unitary/toxic-bert", device=cls_dev)

    nL, nH = model.config.num_hidden_layers, model.config.num_attention_heads
    hd = model.config.hidden_size // nH
    top, rand = load_dart_heads(Path(args.dart_csv), args.n_top, nL, nH, args.seed)
    print(f"Top-{args.n_top} DART heads: {top[:5]}...")
    print(f"Random control heads:      {rand[:5]}...")

    toxic_words, prompts = _load_toxic(args.n_prompts)
    d_tox = toxic_direction(model, tokenizer, _toxic_ids(tokenizer, toxic_words), device)
    mlen = min(args.ppl_max_len, model.config.max_position_embeddings)

    def ev(tag):
        tox, rate = measure_toxicity(model, tokenizer, classifier, prompts, device, args.max_new_tokens)
        ppl = measure_perplexity(model, tokenizer, device, args.ppl_windows, mlen)
        print(f"  [{tag}] mean_tox={tox:.4f} tox_rate={rate:.3f} ppl={ppl:.2f}", flush=True)
        return {"config": tag, "mean_tox": tox, "tox_rate": rate, "ppl": ppl}

    rows = [ev("baseline")]
    with HeadIntervention(model, top, nH, hd):
        rows.append(ev(f"zero_top{args.n_top}"))
    with HeadIntervention(model, rand, nH, hd):
        rows.append(ev(f"zero_random{args.n_top}"))
    flagged_layers = sorted({l for (l, _) in top})
    for lam in args.lambdas:
        if lam == 0.0:
            continue
        with SoftSubtract(model, flagged_layers, d_tox, lam):
            rows.append(ev(f"soft_lambda{lam}"))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "mean_tox", "tox_rate", "ppl"])
        w.writeheader(); w.writerows(rows)
    print(f"Saved {args.out_csv}")
    base = rows[0]
    for r in rows[1:]:
        dtox = 100 * (r["mean_tox"] - base["mean_tox"]) / base["mean_tox"]
        print(f"  {r['config']:>18}: tox {dtox:+.1f}%  ppl {r['ppl']:.1f} (base {base['ppl']:.1f})")


if __name__ == "__main__":
    main()
