"""
Discover toxic attention heads using each method's own lens as the projector.

For each method (unembed, lora, tuned):
  - For each toxic prompt, capture each head's output at attn_out
  - Project through that method's lens to get predicted logits
  - Count how many top-K tokens are in the toxic dictionary
  - Save a toxic_heads_{method}.csv in the format expected by toxicity_ablation.py

Output CSV columns: layer, head_0, head_1, ..., head_11  (counts across all prompts)

Lens loading and the per-head attn_out capture come from the shared
``interp.lenses`` / ``interp.sites`` modules.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from interp import config, lenses, sites
from tox_common import load_toxic_prompts, load_toxic_words, TOXIC_DICT

CKPT_LORA   = config.CKPT_LORA_R64_TOX   # zero-init lens (logit-lens-compatible detection)
CKPT_TUNED  = config.CKPT_TUNED          # full-rank tuned lens (per-layer d×d translator)

NUM_LAYERS = 12
NUM_HEADS = 12
K_TOKENS = 50


@torch.no_grad()
def score_heads(method: str, model, tokenizer, prompts: List[str],
                toxic_words: set, lens, W_U: torch.Tensor,
                device: torch.device) -> np.ndarray:
    """Returns [num_layers, num_heads] toxic token count matrix."""
    counts = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=float)
    collector = sites.HeadOutputCollector(model)

    for p_idx, prompt in enumerate(prompts):
        ids = tokenizer(prompt, max_length=512, truncation=True,
                        return_tensors="pt").input_ids.to(device)
        with collector:
            model(ids)

        for layer in range(NUM_LAYERS):
            if layer not in collector.outputs:
                continue
            head_out = collector.outputs[layer]  # [1, seq, heads, d]
            for head in range(NUM_HEADS):
                act = head_out[:, -1:, head, :]  # [1, 1, d]
                if method == "unembed":
                    logits = lenses.project_unembed(act, W_U)
                else:
                    logits = lenses.project_lens(act, lens, sites.site_id(layer, "attn_out"), device)

                topk_ids = torch.topk(logits, K_TOKENS).indices.tolist()
                topk_toks = [tokenizer.decode([i]).strip().lower() for i in topk_ids]
                counts[layer, head] += sum(1 for t in topk_toks if t in toxic_words)

        if (p_idx + 1) % 25 == 0 or (p_idx + 1) == len(prompts):
            print(f"  [{method}] {p_idx+1}/{len(prompts)} prompts", flush=True)

    return counts


def save_toxic_heads_csv(counts: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"] + [f"head_{h}" for h in range(NUM_HEADS)])
        for layer in range(NUM_LAYERS):
            writer.writerow([layer] + counts[layer].astype(int).tolist())
    print(f"Saved toxic heads → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    choices=["unembed", "lora", "tuned"],
                    default=["unembed", "lora", "tuned"])
    ap.add_argument("--out_dir",   type=Path, required=True)
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--device",    default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    W_U = model.lm_head.weight.detach().cpu()
    toxic_words = load_toxic_words(TOXIC_DICT)
    prompts = load_toxic_prompts(args.n_prompts)
    print(f"Loaded {len(prompts)} toxic prompts, {len(toxic_words)} toxic words")

    ckpt_for = {"lora": CKPT_LORA, "tuned": CKPT_TUNED}
    for method in args.methods:
        print(f"\n=== Scoring heads: {method} ===")
        lens = None if method == "unembed" else lenses.load_lens(
            ckpt_for[method], model, device, verbose=True)

        counts = score_heads(method, model, tokenizer, prompts,
                             toxic_words, lens, W_U, device)

        out_csv = args.out_dir / f"toxic_heads_{method}.csv"
        save_toxic_heads_csv(counts, out_csv)

        flat = [(counts[l, h], l, h) for l in range(NUM_LAYERS) for h in range(NUM_HEADS)]
        flat.sort(reverse=True)
        print("  Top-5 toxic heads: " +
              ", ".join(f"L{l}H{h}={c:.0f}" for c, l, h in flat[:5]))


if __name__ == "__main__":
    main()
