"""
Score every hookable activation site for toxicity, using each method's lens.

For each (layer, site_type) across all 6 per-layer types:
  attn_in, attn_out, resid_mid, mlp_in, mlp_out, resid_post
  - Capture the last-token activation at that site for each toxic prompt
  - Project through the method's lens at the matching site_id
  - Count how many top-K tokens are in the toxic dictionary
  - Average across prompts → toxicity score for that site

Output: toxic_sites_{method}.csv  with columns  site_id, score

Lens loading, the site→module capture and the projection helpers all come from
the shared ``interp.lenses`` / ``interp.sites`` modules — this script only owns the
toxicity scoring.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from interp import config, lenses, sites
from tox_common import load_toxic_prompts, load_toxic_words, TOXIC_DICT

CKPT_LORA  = config.CKPT_LORA_R64_TOX   # zero-init lens (logit-lens-compatible detection)
CKPT_TUNED = config.CKPT_TUNED

NUM_LAYERS = 12
K_TOKENS = 50
SITE_TYPES = sites.SITE_TYPES


@torch.no_grad()
def score_sites(method: str, model, tokenizer, prompts: List[str],
                toxic_words: set, lens, W_U: torch.Tensor,
                device: torch.device) -> Dict[str, float]:
    """Returns {site_id: mean_toxic_count} for all (layer, site_type) pairs."""
    all_site_ids = [sites.site_id(l, st) for l in range(NUM_LAYERS) for st in SITE_TYPES]
    counts = {sid: 0.0 for sid in all_site_ids}
    collector = sites.LastTokenSiteCollector(model)

    for p_idx, prompt in enumerate(prompts):
        ids = tokenizer(prompt, max_length=512, truncation=True,
                        return_tensors="pt").input_ids.to(device)
        with collector:
            model(ids)

        for site_id in all_site_ids:
            act = collector.acts.get(site_id)
            if act is None:
                continue
            if method == "unembed":
                logits = lenses.project_unembed(act, W_U)
            else:
                logits = lenses.project_lens(act, lens, site_id, device)

            topk_ids = torch.topk(logits, K_TOKENS).indices.tolist()
            topk_toks = [tokenizer.decode([i]).strip().lower() for i in topk_ids]
            counts[site_id] += sum(1 for t in topk_toks if t in toxic_words)

        if (p_idx + 1) % 25 == 0 or (p_idx + 1) == len(prompts):
            print(f"  [{method}] {p_idx+1}/{len(prompts)} prompts", flush=True)

    n = len(prompts)
    return {sid: c / n for sid, c in counts.items()}


def save_toxic_sites_csv(scores: Dict[str, float], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["site_id", "score"])
        for sid, score in sorted(scores.items()):
            writer.writerow([sid, f"{score:.4f}"])
    print(f"Saved toxic sites → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    choices=["unembed", "lora", "tuned"],
                    default=["unembed", "lora", "tuned"])
    ap.add_argument("--out_dir",   type=Path, required=True)
    ap.add_argument("--n_prompts", type=int,  default=200)
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
    print(f"Loaded {len(prompts)} prompts, {len(toxic_words)} toxic words")

    ckpt_for = {"lora": CKPT_LORA, "tuned": CKPT_TUNED}
    for method in args.methods:
        print(f"\n=== Scoring sites: {method} ===")
        lens = None if method == "unembed" else lenses.load_lens(
            ckpt_for[method], model, device, verbose=True)

        scores = score_sites(method, model, tokenizer, prompts,
                             toxic_words, lens, W_U, device)

        out_csv = args.out_dir / f"toxic_sites_{method}.csv"
        save_toxic_sites_csv(scores, out_csv)

        top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print("  Top-5: " + ", ".join(f"{s}={c:.2f}" for s, c in top5))


if __name__ == "__main__":
    main()
