"""
Per-head toxicity ablation experiment.

Reuses the selected toxic heads from the existing low-rank run, then compares
subtraction directions at each selected attn_out site:

  unembed       — W_U[toxic_ids].mean(0)                        (baseline)
  lora_r64      — W_U[toxic_ids] @ M.T, forward-direction approx
  tuned         — W_U[toxic_ids] @ M.T from the full-rank tuned lens

Each direction is unit-normalised, then subtracted at scale:
    output -= direction * ||output|| * memory_scale * head_strength

Measures both generation toxicity (unitary/toxic-bert) and WikiText-2 perplexity.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────

from interp import config, lenses, sites
from tox_common import DEFAULT_MEMORY_TEXT, load_toxic_prompts

EXISTING_HEADS    = config.TOXIC_HEADS_CSV
CKPT_LORA_R64     = config.CKPT_LORA_R64_TOX   # zero-init lens for toxicity
CKPT_TUNED        = config.CKPT_TUNED

MEMORY_SCALES  = [0.01, 0.03, 0.1, 0.3]
TOP_FRACTION   = 0.10
MAX_ABLATION   = 1.0

# ── Helpers ───────────────────────────────────────────────────────────────────

@dataclass
class SelectedHead:
    layer:       int
    head:        int
    toxic_count: float
    strength:    float


def read_selected_heads(csv_path: Path, top_fraction: float,
                        max_ablation: float) -> List[SelectedHead]:
    """Re-derive selected heads from the existing toxic_heads.csv."""
    import math
    counts = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            layer = int(row["layer"])
            for k, v in row.items():
                if k.startswith("head_"):
                    counts.append((layer, int(k.removeprefix("head_")), float(v)))
    ranked     = sorted(counts, key=lambda x: x[2], reverse=True)
    n_select   = max(1, math.ceil(len(ranked) * top_fraction))
    selected   = ranked[:n_select]
    max_count  = max(c for _, _, c in selected) or 1.0
    return [
        SelectedHead(layer=l, head=h, toxic_count=c,
                     strength=max_ablation * c / max_count)
        for l, h, c in selected
    ]


# ── Direction builders ─────────────────────────────────────────────────────────

def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp(min=1e-8)


def unembed_directions(model, tokenizer, selected: List[SelectedHead],
                       memory_text: str, device: torch.device
                       ) -> Dict[int, torch.Tensor]:
    toxic_ids = tokenizer.encode(memory_text, add_special_tokens=False)
    W_U       = model.lm_head.weight.detach().float()
    direction = _unit(W_U[sorted(set(toxic_ids))].mean(0))
    return {item.layer: direction.to(device) for item in selected}


def lora_readout_directions(ckpt_dir: Path, model, tokenizer,
                            selected: List[SelectedHead],
                            memory_text: str, device: torch.device
                            ) -> Dict[int, torch.Tensor]:
    """Forward-direction approx: W_U[toxic] @ M.T, normalised per layer."""
    ckpt_path = lenses.resolve_ckpt(ckpt_dir)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    cfg   = ckpt.get("config", {})
    sd    = ckpt["lens_state_dict"]
    r     = int(cfg.get("lora_rank", 64))
    alpha = float(cfg.get("lora_alpha", r))

    toxic_ids   = tokenizer.encode(memory_text, add_special_tokens=False)
    vocab_idx   = sorted(set(toxic_ids))
    W_U         = model.lm_head.weight.detach().float()
    W_subset    = W_U[vocab_idx]                      # [n, d]

    directions: Dict[int, torch.Tensor] = {}
    for layer in sorted({item.layer for item in selected}):
        M_T  = lenses.lora_translator_matrix(sd, sites.site_id(layer, "attn_out"), r, alpha)
        rows = W_subset @ M_T.to(W_subset.device) if M_T is not None else W_subset
        directions[layer] = _unit(rows.mean(0)).to(device)
    print(f"Loaded lora readout directions from {ckpt_path.name}")
    return directions


def tuned_readout_directions(ckpt_dir: Path, model, tokenizer,
                             selected: List[SelectedHead],
                             memory_text: str, device: torch.device
                             ) -> Dict[int, torch.Tensor]:
    """Full-rank tuned-lens read direction: W_U[toxic] @ W_T, normalised per layer."""
    ckpt_path = lenses.resolve_ckpt(ckpt_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd   = ckpt["lens_state_dict"]

    toxic_ids = tokenizer.encode(memory_text, add_special_tokens=False)
    vocab_idx = sorted(set(toxic_ids))
    W_U       = model.lm_head.weight.detach().float()
    W_subset  = W_U[vocab_idx]                         # [n, d]

    directions: Dict[int, torch.Tensor] = {}
    for layer in sorted({item.layer for item in selected}):
        W_T  = lenses.tuned_translator_matrix(sd, sites.site_id(layer, "attn_out"))
        rows = W_subset @ W_T.to(W_subset.device) if W_T is not None else W_subset
        directions[layer] = _unit(rows.mean(0)).to(device)
    print(f"Loaded tuned readout directions from {ckpt_path.name}")
    return directions




# ── Intervention context manager ──────────────────────────────────────────────

def subtract_direction(model: GPT2LMHeadModel,
                       selected: List[SelectedHead],
                       directions: Dict[int, torch.Tensor],
                       memory_scale: float):
    """Subtract direction * ||output|| * memory_scale * strength at each layer's
    attn_out, via the shared site registry (``interp.sites``). Per-layer strength
    is the sum of the selected heads' strengths in that layer.

    Returns the ``edit_at_sites`` context manager, so ``with subtract_direction(...)``
    is unchanged for callers.
    """
    by_layer: Dict[int, float] = {}
    for item in selected:
        by_layer[item.layer] = by_layer.get(item.layer, 0.0) + item.strength

    def make_edit(direction: torch.Tensor, layer_strength: float):
        def edit(h):
            d = direction.to(device=h.device, dtype=h.dtype)
            d = d.view(*([1] * (h.dim() - 1)), h.shape[-1])
            norm = h.detach().float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
            return h - d * norm.to(h.dtype) * (memory_scale * layer_strength)
        return edit

    edits = {sites.site_id(layer, "attn_out"): make_edit(directions[layer], strength)
             for layer, strength in by_layer.items()}
    return sites.edit_at_sites(model, edits)


# ── Measurement ───────────────────────────────────────────────────────────────

def measure_toxicity(model, tokenizer, classifier, prompts: List[str],
                     device: torch.device, max_new_tokens: int = 20) -> dict:
    model.eval()
    toxic_scores, toxic_labels, rows = [], 0, []
    for prompt in tqdm(prompts, desc="toxicity", leave=False):
        inputs = tokenizer(prompt, max_length=900, truncation=True,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text      = tokenizer.decode(out[0], skip_special_tokens=True)
        generated = text[len(prompt):].strip()
        result    = classifier(generated[:450] or " ")[0]
        label, score = result["label"], float(result["score"])
        is_toxic  = label.lower() in {"toxic", "label_1"} or label.upper() == "TOXIC"
        toxic_scores.append(score if is_toxic else 1.0 - score)
        toxic_labels += int(is_toxic)
        rows.append({"prompt": prompt, "generated": generated,
                     "label": label, "score": score})
    return {
        "num_prompts":         len(prompts),
        "toxicity_rate":       toxic_labels / len(prompts),
        "mean_toxicity_score": sum(toxic_scores) / len(toxic_scores),
        "rows":                rows,
    }


def measure_perplexity(model, tokenizer, device: torch.device,
                       stride: int = 512, max_windows: int = 64) -> dict:
    dataset  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    max_len  = model.config.n_positions
    seq_len  = encodings.input_ids.size(1)
    nlls, prev_end, windows = [], 0, 0
    for begin in tqdm(range(0, seq_len, stride), desc="perplexity", leave=False):
        end     = min(begin + max_len, seq_len)
        trg_len = end - prev_end
        ids     = encodings.input_ids[:, begin:end].to(device)
        tgt     = ids.clone()
        tgt[:, :-trg_len] = -100
        with torch.no_grad():
            nlls.append(model(ids, labels=tgt).loss.detach().float().cpu())
        prev_end = end
        windows += 1
        if end == seq_len or windows >= max_windows:
            break
    return {"perplexity": torch.exp(torch.stack(nlls).mean()).item(),
            "windows": windows, "stride": stride}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir",        type=Path, required=True)
    p.add_argument("--toxic_heads_csv",type=Path, default=EXISTING_HEADS)
    p.add_argument("--top_fraction",   type=float, default=TOP_FRACTION)
    p.add_argument("--max_ablation",   type=float, default=MAX_ABLATION)
    p.add_argument("--memory_text",    default=DEFAULT_MEMORY_TEXT)
    p.add_argument("--memory_scales",  type=float, nargs="+", default=MEMORY_SCALES)
    p.add_argument("--n_toxic_prompts",type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--perplexity_windows", type=int, default=64)
    p.add_argument("--perplexity_stride",  type=int, default=512)
    p.add_argument("--methods", nargs="+",
                   choices=["unembed", "lora_r64", "tuned"],
                   default=["unembed", "lora_r64", "tuned"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args    = parse_args()
    device  = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    tokenizer  = GPT2TokenizerFast.from_pretrained("gpt2")
    model      = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    classifier_device = 0 if device.type == "cuda" else -1
    classifier = pipeline("text-classification", model="unitary/toxic-bert",
                          device=classifier_device)

    selected = read_selected_heads(args.toxic_heads_csv, args.top_fraction, args.max_ablation)
    print(f"Using {len(selected)} selected heads from {args.toxic_heads_csv}")

    prompts = load_toxic_prompts(args.n_toxic_prompts)
    print(f"Loaded {len(prompts)} toxic prompts")

    # ── Baseline (no intervention) ──
    print("\n[baseline] no intervention")
    tox = measure_toxicity(model, tokenizer, classifier, prompts, device,
                           args.max_new_tokens)
    ppl = measure_perplexity(model, tokenizer, device,
                              args.perplexity_stride, args.perplexity_windows)
    baseline = {"method": "baseline", "memory_scale": 0.0,
                "toxicity_rate": tox["toxicity_rate"],
                "mean_toxicity_score": tox["mean_toxicity_score"],
                **ppl}
    print(f"  toxicity_rate={tox['toxicity_rate']:.3f}  "
          f"tox_score={tox['mean_toxicity_score']:.4f}  "
          f"ppl={ppl['perplexity']:.2f}")

    summary_rows = [baseline]

    for method in args.methods:
        print(f"\n[method] {method}")

        if method == "unembed":
            directions = unembed_directions(
                model, tokenizer, selected, args.memory_text, device)
        elif method == "lora_r64":
            directions = lora_readout_directions(
                CKPT_LORA_R64, model, tokenizer, selected, args.memory_text, device)
        elif method == "tuned":
            directions = tuned_readout_directions(
                CKPT_TUNED, model, tokenizer, selected, args.memory_text, device)

        method_dir = args.out_dir / method
        method_dir.mkdir(exist_ok=True)

        for scale in args.memory_scales:
            print(f"  scale={scale}", flush=True)
            with subtract_direction(model, selected, directions, scale):
                tox = measure_toxicity(model, tokenizer, classifier, prompts, device,
                                       args.max_new_tokens)
                ppl = measure_perplexity(model, tokenizer, device,
                                          args.perplexity_stride, args.perplexity_windows)

            scale_tag = f"{scale:g}".replace(".", "p")
            scale_dir = method_dir / f"scale_{scale_tag}"
            scale_dir.mkdir(exist_ok=True)

            with open(scale_dir / "toxicity_generations.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(tox["rows"][0].keys()))
                w.writeheader()
                w.writerows(tox.pop("rows"))

            row = {"method": method, "memory_scale": scale,
                   "toxicity_rate": tox["toxicity_rate"],
                   "mean_toxicity_score": tox["mean_toxicity_score"],
                   **ppl}
            with open(scale_dir / "result.json", "w") as f:
                json.dump(row, f, indent=2)
            summary_rows.append(row)

            print(f"    toxicity_rate={tox['toxicity_rate']:.3f}  "
                  f"tox_score={tox['mean_toxicity_score']:.4f}  "
                  f"ppl={ppl['perplexity']:.2f}", flush=True)

    with open(args.out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n[done] results in {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
