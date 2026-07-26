"""Self-contained DART toxic-head localization (GPT-2 / LLaMA 8B / LLaMA 70B).

Mirrors find_toxic_heads_per_method.py but is architecture-aware, loads the
omnilens expanded-hookset lens directly (like lens_tweak_factor_analysis.py),
and cuts corners for speed (this is a "does the lens work at scale" check, not a
rigorous interp study):

  * per-head attn_out via an o_proj pre-hook, split into the query heads
    (GQA only shares K/V — the o_proj input is still num_attention_heads * head_dim),
  * LAST-token head output projected through the lens at ``L{l:02d}.attn_out``,
    batched over heads (one lens call per (prompt, layer)),
  * toxic count via a precomputed set of vocab ids whose decoded token is a
    toxic word (avoids decoding every top-k token of every head).

Output: counts CSV [num_layers x num_heads] per lens, same format the GPT-2 DART
script emits, so downstream cross-lens agreement reuses unchanged.
"""
from __future__ import annotations

import argparse
import csv
import sys
import types
from pathlib import Path

import numpy as np
import torch

K_TOKENS = 50
OMNILENS_DEFAULT = Path("/path/to/project/omnilens/src")


# ── omnilens lens loading (same pattern as the injection script) ─────────────
def _add_omnilens_to_path(path) -> None:
    src = Path(path).expanduser().resolve()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    package_dir = src / "omnilens"
    training_dir = package_dir / "training"
    if package_dir.exists() and "omnilens" not in sys.modules:
        pkg = types.ModuleType("omnilens"); pkg.__path__ = [str(package_dir)]
        sys.modules["omnilens"] = pkg
        if training_dir.exists():
            tr = types.ModuleType("omnilens.training"); tr.__path__ = [str(training_dir)]
            sys.modules["omnilens.training"] = tr


def _resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    cands = sorted(path.glob("lens_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if not cands:
        raise FileNotFoundError(f"No lens_step_*.pt in {path}")
    return cands[-1]


def _load_lens(args, model, device):
    from omnilens.lenses import create_lens
    from omnilens.training.activation_sites import (
        adapt_activation_site_plan_for_model, build_activation_site_plan)
    from omnilens.training.unembed import HFUnembed, get_model_config

    cfg = get_model_config(model)
    plan = build_activation_site_plan(model, num_layers=cfg["num_layers"],
                                      preset=args.activation_site_preset)
    plan = adapt_activation_site_plan_for_model(model, plan)
    kw = dict(layer_ids=plan.site_ids, hidden_size=cfg["hidden_size"], unembed=HFUnembed(model))
    if args.lens_type == "lora":
        kw["r"] = args.lora_rank
        kw["alpha"] = float(args.lora_rank)
    lens = create_lens(args.lens_type, **kw)
    if args.lens_type == "logit":
        # Parameter-free logit lens (ln_f + lm_head); no checkpoint to load.
        print("Using parameter-free logit lens (no checkpoint).")
    else:
        ck = _resolve_checkpoint(Path(args.checkpoint).expanduser())
        state = torch.load(ck, map_location="cpu")
        state = state.get("lens_state_dict", state)
        if hasattr(lens, "load_checkpoint_state_dict"):
            lens.load_checkpoint_state_dict(state)
        else:
            lens.load_state_dict(state, strict=False)
        print(f"Loaded {args.lens_type} lens from {ck}")
    lens.to(device=device).eval()
    return lens, plan


# ── per-head attn output (last token only), GPT-2 + LLaMA ────────────────────
class HeadOutputCollector:
    """Capture each layer's per-head last-token contribution to the residual,
    via a pre-hook on the attention output projection (its input is the
    concatenation of the query-head outputs).

    LLaMA: ``self_attn.o_proj`` is nn.Linear, weight [hidden, heads*head_dim].
    GPT-2: ``attn.c_proj`` is Conv1D (out = x @ W + b), weight [heads*head_dim, hidden].
    """

    def __init__(self, model):
        c = model.config
        self.num_heads = c.num_attention_heads
        self.hidden = c.hidden_size
        self.head_dim = self.hidden // self.num_heads
        self._model = model
        self.outputs: dict[int, torch.Tensor] = {}
        self._handles = []
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self._mods = [(l, b.attn.c_proj) for l, b in enumerate(model.transformer.h)]
            self._W = {l: m.weight.detach().reshape(self.num_heads, self.head_dim, self.hidden)
                       for l, m in self._mods}
            self._ein = "bshd,hde->bshe"
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            self._mods = [(l, b.self_attn.o_proj) for l, b in enumerate(model.model.layers)]
            self._W = {l: m.weight.detach().reshape(self.hidden, self.num_heads, self.head_dim)
                       for l, m in self._mods}
            self._ein = "bshd,ehd->bshe"
        else:
            raise ValueError(f"Unsupported architecture: {type(model)}")

    def _make_hook(self, layer):
        def hook(_mod, inputs):
            merged = inputs[0][:, -1:, :]                       # [b,1,heads*head_dim]
            b = merged.shape[0]
            per_head = merged.reshape(b, 1, self.num_heads, self.head_dim)  # [b,1,h,d]
            W = self._W[layer].to(merged.device, merged.dtype)
            # per-head contribution to residual: [b,1,h,hidden]
            self.outputs[layer] = torch.einsum(self._ein, per_head, W).detach()
        return hook

    def __enter__(self):
        self.outputs = {}
        self._handles = [m.register_forward_pre_hook(self._make_hook(l))
                         for l, m in self._mods]
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles = []


def _build_toxic_id_set(tokenizer, toxic_words: set) -> set:
    """Vocab ids whose single-token decode is a toxic word (compute once)."""
    ids = set()
    vocab = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    for tid in range(vocab):
        tok = tokenizer.decode([tid]).strip().lower()
        if tok in toxic_words:
            ids.add(tid)
    return ids


@torch.no_grad()
def score_heads(model, tokenizer, lens, prompts, toxic_ids, num_layers, num_heads,
                device, autocast_dtype, site_template="L{l:02d}.attn_out"):
    counts = np.zeros((num_layers, num_heads), dtype=float)
    collector = HeadOutputCollector(model)
    for pi, prompt in enumerate(prompts):
        ids = tokenizer(prompt, max_length=512, truncation=True,
                        return_tensors="pt").input_ids.to(device)
        with collector:
            model(ids)
        for layer in range(num_layers):
            if layer not in collector.outputs:
                continue
            heads_in = collector.outputs[layer][0].transpose(0, 1).contiguous().to(device)  # [heads,1,hidden]
            site = site_template.format(l=layer)
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    out = lens(heads_in, layer=site)
            else:
                out = lens(heads_in, layer=site)
            topk = torch.topk(out.logits[:, -1, :].float(), K_TOKENS, dim=-1).indices  # [heads,K]
            for h in range(num_heads):
                counts[layer, h] += sum(1 for t in topk[h].tolist() if t in toxic_ids)
        if (pi + 1) % 25 == 0 or pi + 1 == len(prompts):
            print(f"  {pi+1}/{len(prompts)} prompts", flush=True)
    return counts


def save_counts(counts, num_heads, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer"] + [f"head_{h}" for h in range(num_heads)])
        for l in range(counts.shape[0]):
            w.writerow([l] + counts[l].astype(int).tolist())
    print(f"Saved {path}")


def _load_toxic(n_prompts):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tox_common import load_toxic_words, load_toxic_prompts, TOXIC_DICT
    return load_toxic_words(TOXIC_DICT), load_toxic_prompts(n_prompts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="Lens checkpoint (.pt or dir); not needed for --lens-type logit.")
    ap.add_argument("--lens-type", choices=["logit", "tuned", "lora"], default="lora")
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--activation-site-preset", default="llama_expanded")
    ap.add_argument("--omnilens-src", default=str(OMNILENS_DEFAULT))
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--device-map", default=None,
                    help="e.g. 'auto' to shard a large model across local GPUs (70B).")
    ap.add_argument("--site-template", default="L{l:02d}.attn_out",
                    help="Lens site per layer; use '{l}' for residual-preset lenses "
                         "(decodes head outputs through the layer's block-input "
                         "translator -- documented approximation).")
    args = ap.parse_args()

    _add_omnilens_to_path(args.omnilens_src)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    autocast_dtype = None if dt == torch.float32 else dt

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.device_map:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, low_cpu_mem_usage=True, torch_dtype=dt,
            device_map=args.device_map).eval()
        device = model.lm_head.weight.device
        print(f"Sharded model (device_map={args.device_map}); lens device = {device}")
    else:
        device = torch.device(args.device)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, low_cpu_mem_usage=True, torch_dtype=dt).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lens, _ = _load_lens(args, model, device)

    toxic_words, prompts = _load_toxic(args.n_prompts)
    print(f"{len(prompts)} toxic prompts, {len(toxic_words)} toxic words")
    toxic_ids = _build_toxic_id_set(tokenizer, toxic_words)
    print(f"toxic vocab ids: {len(toxic_ids)}")

    counts = score_heads(model, tokenizer, lens, prompts, toxic_ids,
                         model.config.num_hidden_layers, model.config.num_attention_heads,
                         device, autocast_dtype, site_template=args.site_template)
    save_counts(counts, model.config.num_attention_heads, Path(args.out_csv))
    # quick top-5 head summary
    flat = [(counts[l, h], l, h) for l in range(counts.shape[0]) for h in range(counts.shape[1])]
    flat.sort(reverse=True)
    tot = sum(c for c, _, _ in flat) or 1.0
    print("Top-5 toxic heads:")
    for c, l, h in flat[:5]:
        print(f"  L{l:02d}.H{h:02d}: {int(c)}  ({100*c/tot:.1f}% of signal)")


if __name__ == "__main__":
    main()
