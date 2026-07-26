"""LLaMA-8B whole-model toxicity audit: the 8B version of the GPT-2 all-hooks study.

Two subcommands, mirroring find_toxic_sites_per_method.py + run_hookset_ablation.py
with the GPT-2 registry's hook semantics transplanted to LLaMA:

    site_type    module                     hook   residual meaning
    ---------    -------------------------  -----  --------------------------------
    attn_in      input_layernorm            post   normed input to attention
    attn_out     self_attn.o_proj           post   attention contribution
    resid_mid    post_attention_layernorm   pre    resid_pre + attn_out
    mlp_in       post_attention_layernorm   post   normed input to MLP
    mlp_out      mlp.down_proj              post   MLP contribution
    resid_post   decoder layer              post   hidden_states[L+1]

score:   capture last-token activations at all 6 x n_layers sites over the toxic
         prompts, decode each site through the head+IS lens (batched over prompts),
         count toxic-vocab tokens in the top-50 -> toxic_sites_is.csv (site_id, score).

ablate:  per site type, select the top 10% lens-flagged sites, subtract the
         lens-mapped toxic direction  unit(mean(W_U[toxic] @ M_T_site))  at each,
         with per-site strength proportional to its score, swept over lambda.
         Output summary.csv in the GPT-2 audit's format so the same
         cost-constrained analysis applies. Baseline row copied from the existing
         ToxIn run (identical prompts/settings) or re-measured with --measure-baseline.

Cut for speed, not rigour (this is a "does the density result hold at scale" check).
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from llama_toxin_ablation import measure_toxicity, measure_perplexity, _load_toxic  # noqa: E402
from llama_dart_localization import _add_omnilens_to_path, _load_lens, _build_toxic_id_set  # noqa: E402

K_TOKENS = 50
TOP_FRACTION = 0.10
SITE_TYPES = ["attn_in", "attn_out", "resid_mid", "mlp_in", "mlp_out", "resid_post"]


def resolve_module(model, layer: int, site_type: str):
    b = model.model.layers[layer]
    return {
        "attn_in":    (b.input_layernorm, "post"),
        "attn_out":   (b.self_attn.o_proj, "post"),
        "resid_mid":  (b.post_attention_layernorm, "pre"),
        "mlp_in":     (b.post_attention_layernorm, "post"),
        "mlp_out":    (b.mlp.down_proj, "post"),
        "resid_post": (b, "post"),
    }[site_type]


def module_key(site_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", lambda m: f"_x{ord(m.group(0)):02x}_", site_id)


def _first_tensor(x):
    return x[0] if isinstance(x, tuple) else x


# ── score ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def cmd_score(args, model, tokenizer, lens, device):
    toxic_words, prompts = _load_toxic(args.n_prompts)
    toxic_ids = _build_toxic_id_set(tokenizer, toxic_words)
    print(f"{len(prompts)} prompts, {len(toxic_ids)} toxic vocab ids", flush=True)
    nL = model.config.num_hidden_layers

    store: dict[str, list] = {f"L{l:02d}.{st}": [] for l in range(nL) for st in SITE_TYPES}
    handles = []

    def keep(sid):
        def post(_m, _i, out):
            store[sid].append(_first_tensor(out)[:, -1, :].detach().to("cpu", torch.float32))
        def pre(_m, inputs):
            store[sid].append(inputs[0][:, -1, :].detach().to("cpu", torch.float32))
        return post, pre

    for l in range(nL):
        for st in SITE_TYPES:
            sid = f"L{l:02d}.{st}"
            mod, kind = resolve_module(model, l, st)
            post, pre = keep(sid)
            handles.append(mod.register_forward_hook(post) if kind == "post"
                           else mod.register_forward_pre_hook(pre))

    for pi, p in enumerate(prompts):
        ids = tokenizer(p, max_length=512, truncation=True, return_tensors="pt").input_ids.to(device)
        model(ids)
        if (pi + 1) % 50 == 0:
            print(f"  captured {pi+1}/{len(prompts)}", flush=True)
    for h in handles:
        h.remove()

    rows = []
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    for sid, acts in store.items():
        h = torch.cat([a.unsqueeze(1) for a in acts], 0).to(device)   # [N,1,d]
        with autocast:
            out = lens(h, layer=sid)
        topk = torch.topk(out.logits[:, -1, :].float(), K_TOKENS, dim=-1).indices
        cnt = sum(1 for row in topk.tolist() for t in row if t in toxic_ids)
        rows.append((sid, cnt / len(acts)))
    rows.sort(key=lambda x: x[1], reverse=True)

    out_csv = Path(args.sites_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site_id", "score"])
        w.writerows(rows)
    print(f"Saved {out_csv}; top sites:", flush=True)
    for sid, s in rows[:8]:
        print(f"  {sid:16s} {s:6.2f}")


# ── ablate ───────────────────────────────────────────────────────────────────
def select_sites(sites_csv: Path, site_type: str):
    with open(sites_csv) as f:
        rows = [(r["site_id"], float(r["score"])) for r in csv.DictReader(f)
                if r["site_id"].split(".", 1)[1] == site_type]
    rows.sort(key=lambda x: x[1], reverse=True)
    n = max(1, math.ceil(len(rows) * TOP_FRACTION))
    sel = rows[:n]
    mx = max(s for _, s in sel) or 1.0
    return [(sid, s / mx) for sid, s in sel]


def lens_directions(ckpt_path: Path, model, tokenizer, site_ids, rank, device):
    from tox_common import DEFAULT_MEMORY_TEXT
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("lens_state_dict", sd)
    toxic_ids = sorted(set(tokenizer.encode(DEFAULT_MEMORY_TEXT, add_special_tokens=False)))
    W = model.lm_head.weight.detach().float().cpu()[toxic_ids]      # [n_tox, d]
    dirs = {}
    for sid in site_ids:
        mk = module_key(sid)
        A = sd.get(f"projections.{mk}.lora_A.weight")
        B = sd.get(f"projections.{mk}.lora_B.weight")
        if A is None:
            rows = W
        else:
            M_T = torch.eye(B.shape[0]) + B.float() @ A.float() * (1.0)   # alpha/r = 1
            rows = W @ M_T
        d = rows.mean(0)
        dirs[sid] = (d / d.norm().clamp(min=1e-8)).to(device)
    return dirs


class SiteSubtract:
    """h <- h - (lambda * strength_site) * ||h|| * d_site at each selected site."""

    def __init__(self, model, selected, dirs, lam):
        self._model = model
        self._sel = selected
        self._dirs = dirs
        self._lam = lam
        self._handles = []

    def _edit(self, sid, strength):
        d = self._dirs[sid]
        s = self._lam * strength
        def apply(h):
            nrm = h.norm(dim=-1, keepdim=True)
            return h - s * nrm * d.to(h.dtype).to(h.device)
        def post(_m, _i, out):
            if isinstance(out, tuple):
                return (apply(out[0]),) + out[1:]
            return apply(out)
        def pre(_m, inputs):
            return (apply(inputs[0]),) + inputs[1:]
        return post, pre

    def __enter__(self):
        for sid, strength in self._sel:
            layer, st = int(sid.split(".")[0][1:]), sid.split(".", 1)[1]
            mod, kind = resolve_module(self._model, layer, st)
            post, pre = self._edit(sid, strength)
            self._handles.append(mod.register_forward_hook(post) if kind == "post"
                                 else mod.register_forward_pre_hook(pre))
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles = []


def cmd_ablate(args, model, tokenizer, device):
    from transformers import pipeline
    classifier = pipeline("text-classification", model="unitary/toxic-bert",
                          device=0 if torch.cuda.is_available() else -1)
    _, prompts = _load_toxic(args.n_prompts)
    mlen = min(1024, model.config.max_position_embeddings)

    def ev():
        tox, rate = measure_toxicity(model, tokenizer, classifier, prompts, device, args.max_new_tokens)
        ppl = measure_perplexity(model, tokenizer, device, args.ppl_windows, mlen)
        return tox, rate, ppl

    base_row = None
    if args.baseline_csv:
        with open(args.baseline_csv) as f:
            for r in csv.DictReader(f):
                if r["config"] == "baseline":
                    base_row = (float(r["mean_tox"]), float(r["tox_rate"]), float(r["ppl"]))
        print(f"Baseline reused from {args.baseline_csv}: {base_row}", flush=True)
    if base_row is None:
        print("Measuring baseline...", flush=True)
        base_row = ev()

    for st in args.site_types:
        sel = select_sites(Path(args.sites_csv), st)
        dirs = lens_directions(Path(args.checkpoint), model, tokenizer,
                               [sid for sid, _ in sel], args.lora_rank, device)
        print(f"[{st}] sites: {[sid for sid, _ in sel]}", flush=True)
        rows = [dict(method="baseline", memory_scale=0.0,
                     mean_toxicity_score=base_row[0], tox_rate=base_row[1],
                     perplexity=base_row[2])]
        for lam in args.lambdas:
            with SiteSubtract(model, sel, dirs, lam):
                tox, rate, ppl = ev()
            dtox = 100 * (tox - base_row[0]) / base_row[0]
            print(f"  [{st} lam={lam}] tox={tox:.4f} ({dtox:+.1f}%) ppl={ppl:.2f}", flush=True)
            rows.append(dict(method="lora_r64", memory_scale=lam,
                             mean_toxicity_score=tox, tox_rate=rate, perplexity=ppl))
        out = Path(args.out_root) / st / "summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"Saved {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "ablate"])
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--sites-csv", required=True)
    ap.add_argument("--out-root", default=None, help="ablate: output root dir")
    ap.add_argument("--site-types", nargs="+", default=SITE_TYPES)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--baseline-csv", default=None,
                    help="reuse the baseline row of an existing ToxIn ablation_summary.csv")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--ppl-windows", type=int, default=32)
    ap.add_argument("--n-score-prompts", type=int, default=200)
    ap.add_argument("--omnilens-src",
                    default="/path/to/project/omnilens/src")
    ap.add_argument("--activation-site-preset", default="llama_expanded")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--lens-type", default="lora")
    args = ap.parse_args()

    _add_omnilens_to_path(args.omnilens_src)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if args.cmd == "score":
        args.n_prompts = args.n_score_prompts
        lens, _ = _load_lens(args, model, device)
        cmd_score(args, model, tokenizer, lens, device)
    else:
        cmd_ablate(args, model, tokenizer, device)


if __name__ == "__main__":
    main()
