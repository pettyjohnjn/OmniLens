"""Tweak-factor analysis via trained lenses (tuned / LoRA) instead of TransformerLens.

Replicates experiments/tweak_factor_analysis/injection_tweak_factor_analysis.py but:
  1. Uses HuggingFace + direct PyTorch hooks (no TransformerLens dependency).
  2. Encodes the memory entity via ACTIVATION PATCHING on the RESIDUAL STREAM (resid_mid):
       memory_vec[l] = explicit_resid_mid[l, -1] - implicit_resid_mid[l, -1]
     resid_mid = residual after attention (input to ln_2) — the full model state at that point.
  3. Measures answer probability through the trained lens at each L{l:02d}.resid_mid site.
  4. Handles GPT-2 BPE tokenization: answer tokens are looked up with a leading space.

The tweak_factor t controls interpolation: at t=0, lens sees clean implicit activation;
at t=1, lens sees the explicit prompt's residual state at that layer; t>1 over-injects.

Output format: one CSV per tweak factor, one row per dataset example, with columns
answer_prob_obs, answer_prob_exp, and ans_prob_lens_edit_layer{l} for each layer l.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

OMNILENS_DEFAULT = Path("/path/to/project/omnilens/src")

# ---------------------------------------------------------------------------
# LowRankLens path / import helpers
# ---------------------------------------------------------------------------

def _add_omnilens_to_path(path: str | Path) -> None:
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"LowRankLens src not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    package_dir = src / "omnilens"
    training_dir = package_dir / "training"
    if package_dir.exists():
        if "omnilens" not in sys.modules:
            pkg = types.ModuleType("omnilens")
            pkg.__path__ = [str(package_dir)]
            sys.modules["omnilens"] = pkg
        if "omnilens.training" not in sys.modules and training_dir.exists():
            training = types.ModuleType("omnilens.training")
            training.__path__ = [str(training_dir)]
            sys.modules["omnilens.training"] = training


def _resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(
        path.glob("lens_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No lens_step_*.pt files in {path}")
    return candidates[-1]


def _load_lens(args, model, device):
    from omnilens.lenses import create_lens
    from omnilens.training.activation_sites import (
        adapt_activation_site_plan_for_model,
        build_activation_site_plan,
    )
    from omnilens.training.unembed import HFUnembed, get_model_config

    model_cfg = get_model_config(model)
    plan = build_activation_site_plan(
        model, num_layers=model_cfg["num_layers"], preset=args.activation_site_preset
    )
    plan = adapt_activation_site_plan_for_model(model, plan)

    unembed = HFUnembed(model)
    lens_kwargs: dict = dict(
        layer_ids=plan.site_ids,
        hidden_size=model_cfg["hidden_size"],
        unembed=unembed,
    )
    if args.lens_type == "lora":
        lens_kwargs["r"] = args.lora_rank
        lens_kwargs["alpha"] = float(args.lora_rank)

    if args.lens_type == "logit":
        # parameter-free: frozen ln_f + lm_head, no checkpoint to load
        lens = create_lens("logit", unembed=unembed)
        lens.to(device=device)
        lens.eval()
        print("Using parameter-free logit lens (no checkpoint)")
        return lens, plan

    lens = create_lens(args.lens_type, **lens_kwargs)

    if args.checkpoint is None:
        raise ValueError(f"--checkpoint is required for --lens-type {args.lens_type}")
    ckpt_path = _resolve_checkpoint(Path(args.checkpoint).expanduser())
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("lens_state_dict", ckpt)
    if hasattr(lens, "load_checkpoint_state_dict"):
        lens.load_checkpoint_state_dict(state)
    else:
        lens.load_state_dict(state, strict=False)

    lens.to(device=device)
    lens.eval()
    print(f"Loaded {args.lens_type} lens from {ckpt_path}")
    return lens, plan


# ---------------------------------------------------------------------------
# Activation capture: resid_mid = input to ln_2 = residual after attention
# ---------------------------------------------------------------------------

def _resid_mid_modules(model):
    """Model-aware list of ``(layer_idx, module)`` whose *pre-hook input* is the
    ``resid_mid`` site (the residual after attention, before the norm feeding the MLP).

    This matches omnilens' activation-site plan, where ``resid_mid`` is the *input*
    to ``ln_2`` (GPT-2) / ``post_attention_layernorm`` (LLaMA) — see
    ``activation_sites.py`` (``resid_mid_key = f"{...}_input"``).
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [(l, block.ln_2) for l, block in enumerate(model.transformer.h)]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [(l, block.post_attention_layernorm)
                for l, block in enumerate(model.model.layers)]
    raise ValueError(f"Unsupported architecture for resid_mid capture: {type(model)}")


class _ResidMidCapture:
    """Context manager: captures residual stream AFTER attention (input to the
    MLP-feeding norm) for all layers.

    This is the full model state at each layer — what the ``*_expanded`` presets call
    'resid_mid'. It includes the residual stream from all prior layers plus the current
    attention output, so the lens at this site has complete information. Architecture
    is auto-detected (GPT-2 ``ln_2`` / LLaMA ``post_attention_layernorm``).
    """

    def __init__(self, model):
        self._model = model
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def _make_hook(self, layer_idx: int):
        def pre_hook(module, inp):
            self.activations[layer_idx] = inp[0].detach()
        return pre_hook

    def __enter__(self):
        self.activations = {}
        for l, module in _resid_mid_modules(self._model):
            self._handles.append(
                module.register_forward_pre_hook(self._make_hook(l))
            )
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles = []


def _capture_resid_mids(model, input_ids: torch.Tensor):
    """Clean forward pass — returns per-layer resid_mid tensors (1, seq, d_model) and logits."""
    with _ResidMidCapture(model) as cap:
        with torch.no_grad():
            out = model(input_ids)
    return cap.activations, out.logits


# ---------------------------------------------------------------------------
# Answer token lookup — handles GPT-2 BPE leading-space convention
# ---------------------------------------------------------------------------

def _get_answer_token_id(tokenizer, answer: str) -> int | None:
    """Return the most likely next-token ID for the answer string.

    GPT-2 BPE: words mid-sentence get a leading space (e.g. ' Australia' not 'Australia').
    We try ' {answer}' first, fall back to plain tokenization.
    """
    spaced = tokenizer(" " + answer.strip(), add_special_tokens=False).input_ids
    plain  = tokenizer(answer.strip(),        add_special_tokens=False).input_ids
    ids = spaced if spaced else plain
    return ids[0] if ids else None


# ---------------------------------------------------------------------------
# Lens measurement on interpolated residual
# ---------------------------------------------------------------------------

def _lens_prob_at_patch(
    lens,
    implicit_h: torch.Tensor,   # (1, 1, d_model) — last-token slice of resid_mid
    memory_vec: torch.Tensor,    # (d_model,)
    tweak_factor: float,
    site_id: str,
    answer_id: int,
    autocast_dtype: torch.dtype | None = None,
) -> float:
    patched = implicit_h + memory_vec.reshape(1, 1, -1) * tweak_factor
    with torch.no_grad():
        if autocast_dtype is not None:
            # Match training numerics (AMP bf16) and absorb any lens/model dtype mix.
            with torch.autocast(device_type=patched.device.type, dtype=autocast_dtype):
                lens_out = lens(patched, layer=site_id)
        else:
            lens_out = lens(patched, layer=site_id)
    probs = torch.softmax(lens_out.logits[0, -1, :].float(), dim=-1)
    return float(probs[answer_id].item())


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _attn_out_modules(model):
    """``(layer_idx, module)`` for the attention *output projection*.

    Injecting delta here is the correct way to shift ``resid_mid``: the block computes
    ``resid_mid = attn_out + residual`` and then *re-reads* ``resid_mid`` for BOTH the
    MLP branch and the skip connection. Adding delta to the norm's input (as a naive
    pre-hook would) perturbs only the MLP branch and leaves the skip unpatched.
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [(l, b.attn.c_proj) for l, b in enumerate(model.transformer.h)]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [(l, b.self_attn.o_proj) for l, b in enumerate(model.model.layers)]
    raise ValueError(f"Unsupported architecture for attn-out injection: {type(model)}")


@torch.no_grad()
def causal_profile(model, tokenizer, data, valid, all_impl_outs, all_expl_outs, answer_ids,
                   taus, device, num_layers, batch_size, out_path):
    """CAUSAL injection profile (lens-independent).

    Shift ``resid_mid[l]`` by ``tau * (h_explicit[l] - h_implicit[l])`` at the last token
    -- implemented by adding the delta to the attention output projection, so the whole
    residual stream (MLP branch *and* skip) sees it -- then run the model FORWARD TO THE
    END and read the model's own final P(answer). Batched over examples, left-padded.

    Sanity: at ``tau=1`` and the last layer this must reproduce the explicit prompt's
    final P(answer), since only MLP + final-norm + lm_head remain (all position-wise).

    Also records injection *damage* per (layer, tau): KL(P_injected || P_clean) over
    the full next-token distribution and top-1 retention (fraction of examples whose
    argmax token is unchanged), so gain can be traded off against distortion.
    """
    import csv as _csv

    import numpy as np
    mods = dict(_attn_out_modules(model))
    prompts = [str(data.iloc[i]["obscure_sentence"]) for i in valid]
    ans = [answer_ids[i] for i in valid]
    old_side, old_pad = tokenizer.padding_side, tokenizer.pad_token
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    E = np.zeros((num_layers, len(taus)))
    K = np.zeros((num_layers, len(taus)))   # KL(injected || clean), nats
    R = np.zeros((num_layers, len(taus)))   # top-1 retention vs clean
    n = len(prompts)
    for bs in range(0, n, batch_size):
        idx = valid[bs:bs + batch_size]
        chunk = prompts[bs:bs + batch_size]
        aid = torch.tensor(ans[bs:bs + batch_size], device=device)
        enc = tokenizer(chunk, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        # explicit position_ids so left padding is handled correctly (GPT-2 + LLaMA)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        ar = torch.arange(len(chunk), device=device)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    position_ids=pos)
        log_clean = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        top1_clean = log_clean.argmax(-1)
        for l in range(num_layers):
            if l not in mods:
                continue
            mem = torch.cat([(all_expl_outs[i][l] - all_impl_outs[i][l]) for i in idx], 0).to(device)  # [B,1,d]
            for j, t in enumerate(taus):
                def _hook(_m, _inp, out, mem=mem, t=float(t)):
                    # add delta to the attention output projection -> shifts resid_mid
                    # for both the MLP branch and the residual skip connection
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.clone()
                    h[:, -1:, :] = h[:, -1:, :] + t * mem.to(h.dtype).to(h.device)
                    return ((h,) + tuple(out[1:])) if isinstance(out, tuple) else h
                hd = mods[l].register_forward_hook(_hook)
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                            position_ids=pos)
                hd.remove()
                lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
                p = lp.exp()
                E[l, j] += p[ar, aid].sum().item()
                K[l, j] += (p * (lp - log_clean)).sum(-1).sum().item()
                R[l, j] += (lp.argmax(-1) == top1_clean).float().sum().item()
        print(f"  causal: {min(bs+batch_size,n)}/{n} examples", flush=True)
    E /= n; K /= n; R /= n
    tokenizer.padding_side, tokenizer.pad_token = old_side, old_pad

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["layer"]
                   + [f"causal_P_ans_tau{t}" for t in taus]
                   + [f"causal_KL_tau{t}" for t in taus]
                   + [f"causal_top1keep_tau{t}" for t in taus])
        for l in range(num_layers):
            w.writerow([l] + [f"{E[l, j]:.8f}" for j in range(len(taus))]
                           + [f"{K[l, j]:.6f}" for j in range(len(taus))]
                           + [f"{R[l, j]:.6f}" for j in range(len(taus))])
    print(f"Saved causal profile -> {out_path}")
    for j, t in enumerate(taus):
        b = int(E[:, j].argmax())
        print(f"  tau={t}: best causal injection layer L{b} "
              f"(P={E[b,j]:.5f}, KL={K[b,j]:.3f} nats, top1keep={R[b,j]:.2f})")
    return E


def _load_dataset(name: str, data_root: Path):
    from data.load_data import get_handwritten_data, get_multi_1000
    if name == "hand":
        return get_handwritten_data(str(data_root) + "/")
    if name == "2wmh":
        return get_multi_1000(str(data_root) + "/")
    raise ValueError(f"Unknown dataset: {name!r}")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _add_omnilens_to_path(args.omnilens_src)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                   "float16": torch.float16}[args.dtype]
    # Large models (LLaMA-8B/70B) don't fit single-GPU in fp32; bf16 matches the
    # AMP dtype the lenses were trained under. Non-fp32 → autocast the lens forward.
    autocast_dtype = None if model_dtype == torch.float32 else model_dtype

    if args.device_map:
        # Shard a large model across all local GPUs (e.g. 70B on 4x A100-40GB).
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, low_cpu_mem_usage=True, torch_dtype=model_dtype,
            device_map=args.device_map)
        device = model.lm_head.weight.device  # lens/unembed + all lens math live here
        print(f"Sharded model (device_map={args.device_map}); lens device = {device}")
    else:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, low_cpu_mem_usage=True, torch_dtype=model_dtype)
        model.to(device)
        print(f"Device: {device}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    lens, plan = _load_lens(args, model, device)
    num_layers = model.config.num_hidden_layers

    resid_mid_site_ids = {f"L{l:02d}.resid_mid" for l in range(num_layers)}
    active_sites = [s for s in plan.site_ids if s in resid_mid_site_ids]
    if not active_sites:
        # Residual-preset lenses (e.g. the 8B full-rank tuned reference) carry one
        # translator per layer ("0".."31", trained on block inputs). Read the patched
        # resid_mid through the SAME layer's translator -- a half-block offset
        # (resid_mid(l) = block_input(l) + attn_out(l)), documented in the appendix.
        active_sites = sorted((s for s in plan.site_ids
                               if s.isdigit() and int(s) < num_layers), key=int)
    if not active_sites:
        raise RuntimeError(
            f"No resid_mid or per-layer residual sites in activation plan. "
            f"Available: {plan.site_ids[:10]}"
        )
    print(f"Active readout sites: {len(active_sites)} (e.g. {active_sites[:3]})")

    data_root = REPO_ROOT / "data"
    data = _load_dataset(args.dataset, data_root)
    if args.limit is not None:
        data = data.head(args.limit)
    data = data.reset_index(drop=True)
    print(f"Dataset: {args.dataset}, {len(data)} examples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tweak_factors = list(range(args.min_tweak, args.max_tweak + 1))
    torch.set_grad_enabled(False)

    # Pre-capture all resid_mid activations (2 passes per example), then sweep cheaply
    print("Capturing resid_mid activations (2 passes per example)...")
    all_impl_outs: list[dict[int, torch.Tensor]] = []
    all_expl_outs: list[dict[int, torch.Tensor]] = []
    all_impl_logits: list[torch.Tensor] = []
    all_expl_logits: list[torch.Tensor] = []

    for i, row in data.iterrows():
        impl_ids = tokenizer(str(row["obscure_sentence"]), return_tensors="pt")["input_ids"].to(device)
        expl_ids = tokenizer(str(row["explicit_sentence"]), return_tensors="pt")["input_ids"].to(device)

        impl_outs, impl_logits = _capture_resid_mids(model, impl_ids)
        expl_outs, expl_logits = _capture_resid_mids(model, expl_ids)

        all_impl_outs.append({l: impl_outs[l][:, -1:, :].cpu() for l in impl_outs})
        all_expl_outs.append({l: expl_outs[l][:, -1:, :].cpu() for l in expl_outs})
        all_impl_logits.append(impl_logits[:, -1, :].cpu())
        all_expl_logits.append(expl_logits[:, -1, :].cpu())

        if i % 20 == 0:
            print(f"  captured {i}/{len(data)}")

    print("Activation capture complete. Preparing batched lens sweep...")
    import numpy as np

    # Precompute answer token ids + valid examples (skip untokenizable answers)
    answer_ids = [_get_answer_token_id(tokenizer, str(r["answer"])) for _, r in data.iterrows()]
    valid = [i for i, a in enumerate(answer_ids) if a is not None]
    if not valid:
        raise RuntimeError("No examples with a resolvable answer token.")
    N = len(valid)
    ans_id_t = torch.tensor([answer_ids[i] for i in valid], device=device)  # [N]
    ar = torch.arange(N, device=device)
    print(f"Sweep over {N}/{len(data)} examples with resolvable answers.")

    # CAUSAL profile (lens-independent): where does injection actually help the model?
    if args.causal:
        causal_profile(model, tokenizer, data, valid, all_impl_outs, all_expl_outs,
                       answer_ids, args.causal_taus, device, num_layers,
                       args.causal_batch, output_dir / "causal_profile.csv")
        if args.causal_only:
            print("--causal-only: skipping lens sweep.")
            return

    # Baselines are tweak-independent — compute once, vectorised over examples.
    impl_lg = torch.stack([all_impl_logits[i][0] for i in valid]).float().to(device)  # [N,V]
    expl_lg = torch.stack([all_expl_logits[i][0] for i in valid]).float().to(device)
    obs_prob = torch.softmax(impl_lg, -1)[ar, ans_id_t].cpu().numpy()   # [N]
    exp_prob = torch.softmax(expl_lg, -1)[ar, ans_id_t].cpu().numpy()
    del impl_lg, expl_lg

    # Per-layer example-stacked implicit residual + memory vector (tweak-independent).
    layer_sites = {l: f"L{l:02d}.resid_mid" for l in range(num_layers)
                   if f"L{l:02d}.resid_mid" in plan.site_ids}
    if not layer_sites:  # residual-preset fallback (see active_sites note above)
        layer_sites = {int(s): s for s in plan.site_ids
                       if s.isdigit() and int(s) < num_layers}
    impl_stack = {l: torch.cat([all_impl_outs[i][l] for i in valid], 0).to(device)      # [N,1,d]
                  for l in layer_sites}
    mem_stack = {l: torch.cat([(all_expl_outs[i][l] - all_impl_outs[i][l]) for i in valid], 0).to(device)
                 for l in layer_sites}

    def _fill(col_short_vals):  # map [N] over valid → full-length column (0 elsewhere)
        full = np.zeros(len(data), dtype=float)
        full[valid] = col_short_vals
        return full

    print("Running batched lens sweep...")
    for tweak in tweak_factors:
        cols = {"answer_prob_obs": _fill(obs_prob), "answer_prob_exp": _fill(exp_prob)}
        for l, site_id in layer_sites.items():
            patched = impl_stack[l] + mem_stack[l] * float(tweak)  # [N,1,d]
            with torch.no_grad():
                if autocast_dtype is not None:
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        out = lens(patched, layer=site_id)
                else:
                    out = lens(patched, layer=site_id)
            probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)  # [N,V]
            cols[f"ans_prob_lens_edit_layer{l}"] = _fill(probs[ar, ans_id_t].cpu().numpy())

        data_cp = data.copy()
        for l in range(num_layers):
            data_cp[f"ans_prob_lens_edit_layer{l}"] = 0.0
        for c, v in cols.items():
            data_cp[c] = v

        before = data_cp["answer_prob_obs"].mean()
        best_l, best_v = max(((l, data_cp[f"ans_prob_lens_edit_layer{l}"].mean())
                              for l in layer_sites), key=lambda x: x[1])
        print(f"=== tweak={tweak}: obs={before:.4f} | best layer {best_l} "
              f"lens_patch={best_v:.4f} ({(best_v-before)/max(before,1e-9)*100:+.1f}%)")

        out_path = output_dir / f"tweak_{tweak}.csv"
        data_cp.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--omnilens-src", default=str(OMNILENS_DEFAULT))
    p.add_argument("--model-name", default="gpt2")
    p.add_argument("--checkpoint", default=None,
                   help="Path to checkpoint directory or .pt file (not needed for --lens-type logit)")
    p.add_argument("--lens-type", choices=["logit", "tuned", "lora"], default="lora")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"],
                   default="float32",
                   help="Model dtype; use bfloat16 for LLaMA-8B/70B (matches AMP training).")
    p.add_argument("--activation-site-preset",
                   choices=["residual", "gpt2_expanded", "llama_expanded"],
                   default="gpt2_expanded")
    p.add_argument("--dataset", choices=["hand", "2wmh"], required=True)
    p.add_argument("--max-tweak", type=int, default=15)
    p.add_argument("--min-tweak", type=int, default=1,
                   help="Set 0 to also record the clean implicit readout (tau=0); "
                        "tau=1 is exactly the explicit readout.")
    p.add_argument("--causal", action="store_true",
                   help="Also compute the lens-independent causal injection profile "
                        "(patch resid_mid[l], full forward, model's own final P(answer)).")
    p.add_argument("--causal-only", action="store_true", help="Skip the lens sweep.")
    p.add_argument("--causal-taus", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--causal-batch", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="Cap dataset at N examples (debug)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--device-map", default=None,
                   help="e.g. 'auto' to shard a large model across all local GPUs (70B).")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
