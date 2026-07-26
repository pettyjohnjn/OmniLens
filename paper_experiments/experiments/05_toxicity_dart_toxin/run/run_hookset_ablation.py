"""
Toxicity ablation over a configurable hook set.

Loads a toxic_sites_{selector}.csv produced by find_toxic_sites_per_method.py,
optionally filters to one site_type via --site_type_filter, selects top-fraction
sites, then subtracts lens-derived directions at each selected hook point.

Supported site types: attn_in, attn_out, resid_mid, mlp_in, mlp_out, resid_post

Methods for the ablation direction:
  unembed      — W_U[toxic_ids].mean(0)  (no lens)
  lora_r64     — W_U[toxic] @ M_T  where M_T = I + B@A*(alpha/r)
  tuned        — W_U[toxic] @ W_T  (full-rank translator weight)

Usage:
  python run_hookset_ablation.py \
      --out_dir <path> \
      --toxic_sites_csv <path/toxic_sites_lora.csv> \
      --site_type_filter attn_out \
      --methods lora_r64 \
      --memory_scales 0.05 0.10 0.17 0.25 0.35
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline

from interp import config, lenses, sites
from tox_common import DEFAULT_MEMORY_TEXT

CKPT_LORA_R64     = config.CKPT_LORA_R64_TOX   # zero-init lens for toxicity
CKPT_TUNED        = config.CKPT_TUNED

TOP_FRACTION = 0.10
MAX_ABLATION = 1.0


@dataclass
class SelectedSite:
    site_id:     str
    layer:       int
    site_type:   str
    toxic_count: float
    strength:    float


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp(min=1e-8)


# ── Site selection ─────────────────────────────────────────────────────────────

def read_selected_sites(csv_path: Path, top_fraction: float,
                        max_ablation: float,
                        site_type_filter: Optional[str] = None) -> List[SelectedSite]:
    with open(csv_path) as f:
        rows = [(r["site_id"], float(r["score"])) for r in csv.DictReader(f)]

    # Optionally restrict to one site type (e.g. "attn_out")
    if site_type_filter:
        rows = [(sid, s) for sid, s in rows if sid.split(".", 1)[1] == site_type_filter]

    ranked    = sorted(rows, key=lambda x: x[1], reverse=True)
    n_select  = max(1, math.ceil(len(ranked) * top_fraction))
    selected  = ranked[:n_select]
    max_score = max(s for _, s in selected) or 1.0

    result = []
    for site_id, score in selected:
        # site_id format: "L05.mlp_out"
        parts     = site_id.split(".", 1)
        layer     = int(parts[0][1:])  # "L05" → 5
        site_type = parts[1]           # "mlp_out", "resid_mid", etc.
        result.append(SelectedSite(
            site_id     = site_id,
            layer       = layer,
            site_type   = site_type,
            toxic_count = score,
            strength    = max_ablation * score / max_score,
        ))
    return result


# ── Direction builders ─────────────────────────────────────────────────────────

def unembed_directions(model, tokenizer, selected: List[SelectedSite],
                       memory_text: str, device: torch.device) -> Dict[str, torch.Tensor]:
    toxic_ids = tokenizer.encode(memory_text, add_special_tokens=False)
    W_U       = model.lm_head.weight.detach().float()
    direction = _unit(W_U[sorted(set(toxic_ids))].mean(0))
    return {site.site_id: direction.to(device) for site in selected}


def lora_readout_directions(ckpt_dir: Path, model, tokenizer,
                            selected: List[SelectedSite],
                            memory_text: str, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt_path = lenses.resolve_ckpt(ckpt_dir)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    cfg   = ckpt.get("config", {})
    sd    = ckpt["lens_state_dict"]
    r     = int(cfg.get("lora_rank", 64))
    alpha = float(cfg.get("lora_alpha", r))

    toxic_ids = tokenizer.encode(memory_text, add_special_tokens=False)
    W_U       = model.lm_head.weight.detach().float()
    W_subset  = W_U[sorted(set(toxic_ids))]

    directions = {}
    for site in selected:
        M_T  = lenses.lora_translator_matrix(sd, site.site_id, r, alpha)
        rows = W_subset @ M_T.to(W_subset.device) if M_T is not None else W_subset
        directions[site.site_id] = _unit(rows.mean(0)).to(device)
    print(f"Loaded lora readout directions from {ckpt_path.name}")
    return directions


def tuned_readout_directions(ckpt_dir: Path, model, tokenizer,
                             selected: List[SelectedSite],
                             memory_text: str, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt_path = lenses.resolve_ckpt(ckpt_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd   = ckpt["lens_state_dict"]

    toxic_ids = tokenizer.encode(memory_text, add_special_tokens=False)
    W_U       = model.lm_head.weight.detach().float()
    W_subset  = W_U[sorted(set(toxic_ids))]

    directions = {}
    for site in selected:
        W_T  = lenses.tuned_translator_matrix(sd, site.site_id)
        rows = W_subset @ W_T.to(W_subset.device) if W_T is not None else W_subset
        directions[site.site_id] = _unit(rows.mean(0)).to(device)
    print(f"Loaded tuned readout directions from {ckpt_path.name}")
    return directions




# ── Ablation hook ──────────────────────────────────────────────────────────────

def subtract_at_sites(model, selected: List[SelectedSite],
                      directions: Dict[str, torch.Tensor], scale: float):
    """Norm-relative subtraction at each selected site, via the shared site
    registry (``interp.sites``). Each site's activation ``h`` (module input for
    ``resid_mid``, module output otherwise — the registry decides) becomes

        h ← h − dir · ‖h‖ · scale · strength

    Returns the ``edit_at_sites`` context manager, so ``with subtract_at_sites(...)``
    is unchanged for callers.
    """
    def make_edit(direction: torch.Tensor, strength: float):
        def edit(h):
            d = direction.to(h.device)
            n = h.norm(dim=-1, keepdim=True)
            return h - d * n * scale * strength
        return edit

    edits = {site.site_id: make_edit(directions[site.site_id], site.strength)
             for site in selected if site.site_id in directions}
    return sites.edit_at_sites(model, edits)


# ── Measurement helpers (identical to toxicity_ablation.py) ─────────────

def measure_toxicity(model, tokenizer, classifier, prompts, device, max_new_tokens):
    model.eval()
    rows, scores = [], []
    for prompt in tqdm(prompts, desc="toxicity"):
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=200).input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        result  = classifier(gen[:512] or ".", truncation=True)[0]
        is_tox  = result["label"].lower() == "toxic"
        score   = result["score"] if is_tox else 1.0 - result["score"]
        rows.append({"prompt": prompt[:80], "generation": gen[:120],
                     "toxic": int(is_tox), "score": score})
        scores.append(score)
    return {"toxicity_rate": sum(r["toxic"] for r in rows) / len(rows),
            "mean_toxicity_score": sum(scores) / len(scores), "rows": rows}


@torch.no_grad()
def measure_perplexity(model, tokenizer, device, stride, n_windows):
    dataset  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text     = "\n\n".join(dataset["text"])
    enc      = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]
    max_len  = model.config.n_positions
    nlls, count = [], 0
    for begin in range(0, len(input_ids) - max_len, stride):
        chunk = input_ids[begin: begin + max_len].unsqueeze(0).to(device)
        with torch.no_grad():
            loss = model(chunk, labels=chunk).loss
        nlls.append(loss.item())
        count += 1
        if count >= n_windows:
            break
    import math
    ppl = math.exp(sum(nlls) / len(nlls))
    return {"perplexity": ppl, "windows": count, "stride": stride}


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_SITE_TYPES = ["attn_in", "attn_out", "resid_mid", "mlp_in", "mlp_out", "resid_post"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir",          type=Path, required=True)
    p.add_argument("--toxic_sites_csv",  type=Path, required=True)
    p.add_argument("--site_type_filter", choices=ALL_SITE_TYPES, default=None,
                   help="If set, restrict ablation to this site type only.")
    p.add_argument("--memory_text",      default=DEFAULT_MEMORY_TEXT)
    p.add_argument("--memory_scales",    type=float, nargs="+", default=[0.05, 0.10, 0.17, 0.25, 0.35])
    p.add_argument("--top_fraction",     type=float, default=TOP_FRACTION)
    p.add_argument("--n_toxic_prompts",  type=int,   default=100)
    p.add_argument("--max_new_tokens",   type=int,   default=20)
    p.add_argument("--perplexity_windows", type=int, default=64)
    p.add_argument("--perplexity_stride", type=int,  default=512)
    p.add_argument("--methods", nargs="+",
                   choices=["unembed", "lora_r64", "tuned"],
                   default=["unembed", "lora_r64", "tuned"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model     = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    classifier = pipeline("text-classification",
                          model="unitary/toxic-bert", device=device)

    selected = read_selected_sites(args.toxic_sites_csv, args.top_fraction, MAX_ABLATION,
                                   site_type_filter=args.site_type_filter)
    filter_tag = args.site_type_filter or "all"
    print(f"Selected {len(selected)} sites (site_type_filter={filter_tag}):")
    for s in selected:
        print(f"  {s.site_id}  score={s.toxic_count:.3f}  strength={s.strength:.3f}")

    dataset = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="train")
    prompts = []
    for row in dataset:
        if row.get("label") == 1 and isinstance(row.get("comment_text"), str):
            prompts.append(row["comment_text"].strip())
        if len(prompts) >= args.n_toxic_prompts:
            break

    # Baseline (no ablation)
    print("\n[baseline]")
    base_tox = measure_toxicity(model, tokenizer, classifier, prompts, device,
                                args.max_new_tokens)
    base_ppl = measure_perplexity(model, tokenizer, device,
                                  args.perplexity_stride, args.perplexity_windows)
    baseline = {"method": "baseline", "memory_scale": 0.0,
                "toxicity_rate": base_tox["toxicity_rate"],
                "mean_toxicity_score": base_tox["mean_toxicity_score"], **base_ppl}
    print(f"  tox={baseline['mean_toxicity_score']:.4f}  ppl={baseline['perplexity']:.2f}")

    summary_rows = [baseline]

    for method in args.methods:
        print(f"\n[method] {method}")
        if method == "unembed":
            directions = unembed_directions(model, tokenizer, selected, args.memory_text, device)
        elif method == "lora_r64":
            directions = lora_readout_directions(CKPT_LORA_R64, model, tokenizer,
                                                 selected, args.memory_text, device)
        elif method == "tuned":
            directions = tuned_readout_directions(CKPT_TUNED, model, tokenizer,
                                                  selected, args.memory_text, device)

        method_dir = args.out_dir / method
        method_dir.mkdir(exist_ok=True)

        for scale in args.memory_scales:
            print(f"  scale={scale}", flush=True)
            with subtract_at_sites(model, selected, directions, scale):
                tox = measure_toxicity(model, tokenizer, classifier, prompts, device,
                                       args.max_new_tokens)
                ppl = measure_perplexity(model, tokenizer, device,
                                         args.perplexity_stride, args.perplexity_windows)

            scale_tag = f"{scale:g}".replace(".", "p")
            scale_dir = method_dir / f"scale_{scale_tag}"
            scale_dir.mkdir(exist_ok=True)

            with open(scale_dir / "toxicity_generations.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(tox["rows"][0].keys()))
                w.writeheader(); w.writerows(tox.pop("rows"))

            row = {"method": method, "memory_scale": scale,
                   "toxicity_rate": tox["toxicity_rate"],
                   "mean_toxicity_score": tox["mean_toxicity_score"], **ppl}
            with open(scale_dir / "result.json", "w") as f:
                json.dump(row, f, indent=2)
            summary_rows.append(row)
            print(f"    tox={tox['mean_toxicity_score']:.4f}  ppl={ppl['perplexity']:.2f}")

    with open(args.out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"\n[done] {args.out_dir}/summary.csv")


if __name__ == "__main__":
    main()
