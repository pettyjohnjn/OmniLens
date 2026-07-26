#!/usr/bin/env python3
"""Is the full-rank tuned lens functionally low rank?

Three curves on one protocol, all measured against the same teacher pass:

  trained    trained low-rank lens at rank r (what the paper's rank ablation shows)
  truncated  the full-rank tuned lens with its Delta truncated to rank r by SVD
  random     the same full-rank Delta projected onto a random r-dim subspace

The spectral study (omnilens/analysis/intrinsic_dimensionality) found
Delta is NOT low rank as a matrix: rank 64 holds only 58.5% of its Frobenius
energy at GPT-2 and 35.4% at 8B. That is a statement about matrix reconstruction
in all of R^d. This script asks the functional question instead -- how much KL
does a rank-r lens actually lose -- with no linearization and no choice of norm.

Lens forward, from tuned_lens.py:136-141 and lowrank_lens.py:83-90:
    logits = W_U( eta( h + h @ Delta.T + b ) )
so a variant is fully specified by (Delta, b); no lens object is instantiated.
eta and W_U come from the frozen model, identical for every variant.

Env: omnilens. Modeled on convergence_study/eval_llama70b_convergence.py.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

RANKS = [1, 4, 8, 16, 32, 64, 128, 256, 384]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="gpt2")
    p.add_argument("--ckpt-root", default="/path/to/project/omnilens/src/checkpoints/gpt2")
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "data"))
    p.add_argument("--tokens", type=int, default=131072)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--step", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default=None, help="comma-separated variant names (smoke tests)")
    p.add_argument("--out-name", default="gpt2_rank_truncation.csv")
    p.add_argument("--corpus", default="pile", choices=["pile", "wikitext2"])
    return p.parse_args()


def load_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd.get("lens_state_dict", sd), sd.get("config", {})


def tuned_deltas(path):
    """Full-rank tuned lens -> {layer: (Delta, bias)}. Stored weight IS Delta."""
    sd, _ = load_state(path)
    out = {}
    for k, v in sd.items():
        if k.startswith("translators.") and k.endswith(".weight"):
            i = int(k.split(".")[1])
            b = sd.get(f"translators.{i}.bias")
            out[i] = (v.float(), None if b is None else b.float())
    return out


def lora_deltas(path):
    """low-rank lens -> {layer: (Delta_eff, bias)}, Delta_eff = (alpha/r) B @ A."""
    sd, cfg = load_state(path)
    r = int(cfg["lora_rank"])
    alpha = cfg.get("lora_alpha") or float(r)
    scaling = float(alpha) / r
    out = {}
    for k in sd:
        if k.endswith("lora_A.weight"):
            pre = k[: -len("lora_A.weight")]
            i = int([p for p in pre.split(".") if p.isdigit()][0])
            A = sd[pre + "lora_A.weight"].float()          # [r, d]
            B = sd[pre + "lora_B.weight"].float()          # [d, r]
            b = sd.get(pre + "bias")
            out[i] = (scaling * (B @ A), None if b is None else b.float())
    return out


def truncate(delta, r):
    U, S, Vh = torch.linalg.svd(delta.double(), full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r], S


def random_project(delta, r, gen):
    """Project Delta onto a random r-dim output subspace (control for 'is the
    top-r singular subspace special, or would any r directions do?')."""
    d = delta.shape[0]
    Q, _ = torch.linalg.qr(torch.randn(d, r, generator=gen, dtype=torch.float64))
    return Q @ (Q.T @ delta.double())


PILE_VAL = ("/path/to/hf_cache/hub/"
            "datasets--monology--pile-uncopyrighted/snapshots/"
            "3be90335b66f24456a5d6659d9c8d208c0357119/val.jsonl.zst")


def load_windows(tokenizer, n_tokens, seq_len, corpus):
    """Pile val is the held-out split of the training distribution, so its KL is
    comparable to the paper's rank-ablation table; wikitext-2 is out-of-domain
    for these lenses and reads ~4x higher."""
    from datasets import load_dataset
    if corpus == "pile":
        ds = load_dataset("json", data_files={"val": PILE_VAL}, split="val", streaming=True)
        # shuffle: taking documents in file order samples one narrow slice of the
        # Pile's mixture and inflates KL for every variant alike
        ds = ds.shuffle(seed=0, buffer_size=10000)
        ids, need = [], n_tokens + seq_len
        for rec in ds:
            ids.extend(tokenizer(rec["text"]).input_ids)
            if len(ids) >= need:
                break
        ids = torch.tensor(ids[:need])
        name = "pile-uncopyrighted:val"
    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
        name = "wikitext-2-raw-v1:test"
    n = n_tokens // seq_len
    if len(ids) < n * seq_len:
        n = len(ids) // seq_len
        print(f"[warn] corpus holds only {n} windows", flush=True)
    return [ids[i * seq_len:(i + 1) * seq_len] for i in range(n)], name


@torch.no_grad()
def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.ckpt_root)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32).to(dev).eval()
    ln_f, W_U = model.transformer.ln_f, model.lm_head

    tuned_p = root / f"tuned/kl/residual/seed0/lens_step_{args.step}.pt"
    base = tuned_deltas(tuned_p)
    print(f"[main] full-rank tuned lens: {len(base)} translators from {tuned_p.name}")

    gen = torch.Generator().manual_seed(args.seed)
    variants = {}                                    # name -> {layer: (Delta, bias)}
    variants["full_rank"] = base
    for r in RANKS:
        variants[f"truncated_r{r}"] = {i: (truncate(D, r)[0].float(), b) for i, (D, b) in base.items()}
        variants[f"random_r{r}"] = {i: (random_project(D, r, gen).float(), b) for i, (D, b) in base.items()}
        lp = root / f"lora-r{r}/kl/residual/init-default/seed0/lens_step_{args.step}.pt"
        if lp.exists():
            variants[f"trained_r{r}"] = lora_deltas(lp)
        else:
            print(f"[warn] missing {lp}")
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        variants = {k: v for k, v in variants.items() if k in keep}
    print(f"[main] {len(variants)} variants: {sorted(variants)}")

    # move to device once
    variants = {n: {i: (D.to(dev), None if b is None else b.to(dev)) for i, (D, b) in v.items()}
                for n, v in variants.items()}

    windows, corpus = load_windows(tok, args.tokens, args.seq_len, args.corpus)
    print(f"[main] {len(windows)} windows x {args.seq_len} from {corpus}")

    layers = sorted(base)
    kl_sum = {n: {i: 0.0 for i in layers} for n in variants}
    n_tok = 0
    for wi, win in enumerate(windows):
        ids = win.unsqueeze(0).to(dev)
        out = model(ids, output_hidden_states=True)
        t_logp = F.log_softmax(out.logits.float(), dim=-1)
        t_p = t_logp.exp()
        for name, spec in variants.items():
            for i in layers:
                D, b = spec[i]
                h = out.hidden_states[i].float()          # translator i reads hidden_state i
                z = h + h @ D.T + (0.0 if b is None else b)
                lg = W_U(ln_f(z))
                kl = (t_p * (t_logp - F.log_softmax(lg.float(), dim=-1))).sum(-1).mean().item()
                kl_sum[name][i] += kl * ids.shape[1]
        n_tok += ids.shape[1]
        if (wi + 1) % 16 == 0 or wi == len(windows) - 1:
            print(f"[main] window {wi+1}/{len(windows)}")

    csv_p = out_dir / args.out_name
    with open(csv_p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["variant", "family", "rank", "layer", "kl"])
        for name, per in kl_sum.items():
            fam, _, rs = name.partition("_r")
            rank = "full" if name == "full_rank" else rs
            for i in layers:
                w.writerow([name, fam, rank, i, f"{per[i] / n_tok:.6f}"])
    (out_dir / "meta.json").write_text(json.dumps({
        "corpus": corpus, "tokens": n_tok, "seq_len": args.seq_len, "step": args.step,
        "model": args.model_name, "dtype": "fp32 teacher and lens", "seed": args.seed,
        "note": "trained = low-rank lens at rank r; truncated = full-rank Delta cut to rank r by SVD; "
                "random = same Delta projected onto a random r-dim subspace",
    }, indent=2))
    print(f"[main] wrote {csv_p}")

    fin = max(layers)
    print(f"\n{'variant':>16} {'final-layer KL':>15} {'mean KL':>10}")
    for name in sorted(kl_sum):
        per = {i: kl_sum[name][i] / n_tok for i in layers}
        print(f"{name:>16} {per[fin]:15.4f} {sum(per.values())/len(per):10.4f}")


if __name__ == "__main__":
    main()
