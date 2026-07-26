"""Causal basis extraction (CBE) per lens variant, after Belrose et al. 2023.

The remaining quantitative experiment from the tuned-lens paper: for each lens
and layer, find the k orthonormal directions of the residual stream whose mean
ablation most changes the LENS output (L-BFGS with deflation, "energy" = the
expected KL increase; algorithm adapted from tuned_lens/causal/subspaces.py in
the EleutherAI repo), then ablate each direction in the MODEL forward pass and
measure the KL of the perturbed final distribution against the clean one.

A lens is causally faithful when the directions it considers influential are
influential in the model: we report the per-layer Spearman correlation between
lens energy and model KL, and the mean model KL of each lens's top directions
against a random orthonormal basis (control). The paper's claim to reproduce is
that trained-lens bases transfer causally; ours matches the full-rank tuned
reference if its correlations and transfer KLs are comparable.

Text: WikiText-2 (train). Extraction uses P last-token-agnostic positions
sampled from N sequences; evaluation ablates every position of a held-out
batch. Reuses the model/lens setup of injection_detection.py.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/path/to/project/interp/src")
from injection_detection import (MODEL_SPECS, _setup_model_and_lenses,  # noqa: E402
                                 trajectory_sites)
from interp.lenses import lora_translator_matrix, tuned_translator_matrix  # noqa: E402


# ── subspace removal (verbatim math from tuned-lens repo, trimmed) ───────────

def remove_subspace(u: torch.Tensor, A: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Remove information in ``u`` along span(A) (A orthonormal, 1D or 2D)."""
    if A.ndim == 1:
        A = A[..., None]
    proj = A @ A.mT
    if mode == "zero":
        dummy = -u
    elif mode == "mean":
        dummy = u.flatten(0, -2).mean(0) - u
    else:
        raise ValueError(mode)
    return u + torch.einsum("ij,...j->...i", proj, dummy)


def _blocks(model):
    if hasattr(model, "transformer"):
        return model.transformer.h
    return model.model.layers


@contextmanager
def ablate_at_block(model, hs_index: int, v: torch.Tensor):
    """Mean-ablate the span of ``v`` from the output of block hs_index-1."""
    block = _blocks(model)[hs_index - 1]

    def hook(_, __, out):
        h = out[0] if isinstance(out, tuple) else out
        h_ = remove_subspace(h.float(), v.to(h.device).float()).to(h.dtype)
        return (h_, *out[1:]) if isinstance(out, tuple) else h_

    hd = block.register_forward_hook(hook)
    try:
        yield
    finally:
        hd.remove()


# ── data ─────────────────────────────────────────────────────────────────────

def wikitext_batches(tokenizer, n_seq: int, seq_len: int, seed: int):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(r["text"] for r in ds if len(r["text"]) > 200)
    ids = tokenizer.encode(text, add_special_tokens=False)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(ids) - seq_len - 1, n_seq)
    return torch.tensor([ids[s:s + seq_len] for s in starts], dtype=torch.long)


# ── extraction (their L-BFGS loop, against the omnilens call convention) ─────

def extract_basis(lens, site: str, H: torch.Tensor, k: int, max_iter: int,
                  init: torch.Tensor, autocast_dtype):
    """CausalBasis for one lens at one site. H: [P, d] float32 on lens device.

    Returns (energies [k], vectors [d, k]); energy = E[KL(p_clean || p_ablated)]
    of the LENS readout under mean ablation of the direction.
    """
    device, d = H.device, H.shape[-1]
    eye = torch.eye(d, device=device)

    def lens_logits(h):  # h: [P, d] -> [P, V]
        if autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                return lens(h.unsqueeze(0), layer=site).logits[0].float()
        return lens(h.unsqueeze(0), layer=site).logits[0].float()

    with torch.no_grad():
        log_p = lens_logits(H).log_softmax(-1)
        p = log_p.exp()

    energies = torch.zeros(k, device=device)
    vectors = init[:, :k].clone().to(device)

    for j in range(k):
        proj = eye - vectors[:, :j] @ vectors[:, :j].T if j else eye

        def project(x):
            x = proj @ x
            return x / (x.norm() + torch.finfo(x.dtype).eps)

        vectors[:, j] = project(vectors[:, j])
        v = torch.nn.Parameter(vectors[:, j].clone())
        opt = torch.optim.LBFGS([v], line_search_fn="strong_wolfe", max_iter=max_iter)
        nfev, last_energy, energy_delta = 0, torch.tensor(0.0, device=device), None

        def closure():
            nonlocal nfev, last_energy, energy_delta
            nfev += 1
            opt.zero_grad(set_to_none=False)
            h_ = remove_subspace(H, project(v))
            log_q = lens_logits(h_).log_softmax(-1)
            loss = -torch.sum(p * (log_p - log_q), dim=-1).mean()
            loss.backward()
            new_energy = -loss.detach()
            energy_delta = new_energy - last_energy
            last_energy = new_energy
            if not loss.isfinite():
                loss = torch.tensor(0.0, device=device)
                opt.zero_grad(set_to_none=False)
            return loss

        while nfev < max_iter:
            opt.step(closure)
            v.data = project(v.data)
            if abs(energy_delta / (last_energy + 1e-12)) < 1e-4:
                break

        vectors[:, j] = project(v.data)
        energies[j] = last_energy

    order = energies.argsort(descending=True)
    return energies[order].detach(), vectors[:, order].detach()


def svd_init(model, lens_name: str, ckpt_path, site: str, k: int, device):
    """Init directions: top left singular vectors of (M_T @ W_U^T), their init."""
    W_U = model.lm_head.weight.detach().float()  # [V, d]
    d = W_U.shape[1]
    M = None
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("lens_state_dict", ckpt)
        cfg = ckpt.get("config", {})
        if cfg.get("lens_type", "lora") == "lora":
            r = int(cfg.get("lora_rank", 64))
            M = lora_translator_matrix(state, site, r, float(cfg.get("lora_alpha", r)))
        else:
            M = tuned_translator_matrix(state, site)
    X = W_U.T.to(device) if M is None else (M.to(device) @ W_U.T.to(device))
    u, _, _ = torch.svd_lowrank(X, q=min(k + 8, d))
    return u[:, :k].contiguous()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(MODEL_SPECS), required=True)
    p.add_argument("--layers", default=None, help="comma list of hs indices")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--max-iter", type=int, default=40)
    p.add_argument("--pos", type=int, default=1024, help="extraction positions")
    p.add_argument("--eval-seq", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    (model, tokenizer, lenses, n_layers, lens_device, input_device,
     autocast_dtype) = _setup_model_and_lenses(args)
    spec = MODEL_SPECS[args.model]

    if args.layers:
        hs_list = [int(x) for x in args.layers.split(",")]
    else:
        step = max(2, (n_layers + 1) // 8)
        hs_list = list(range(step, n_layers, step))

    # site id per lens per hs index
    site_of = {name: {hi: site for hi, site in sites}
               for name, (_, sites) in lenses.items()}

    # hidden states: one forward over extraction + eval batches
    n_ext_seq = max(args.pos // args.seq_len, 4)
    ids = wikitext_batches(tokenizer, n_ext_seq + args.eval_seq,
                           args.seq_len, args.seed).to(input_device)
    ext_ids, ev_ids = ids[:n_ext_seq], ids[n_ext_seq:]
    rng = np.random.default_rng(args.seed)

    H_ext = {}
    with torch.no_grad():
        out = model(ext_ids, output_hidden_states=True)
        flat_idx = rng.permutation(n_ext_seq * args.seq_len)[: args.pos]
        for hi in hs_list:
            h = out.hidden_states[hi].to(lens_device).float()
            H_ext[hi] = h.flatten(0, 1)[flat_idx].contiguous()
        del out
        clean = model(ev_ids).logits.float()
        log_clean = clean.log_softmax(-1).to(lens_device)
        p_clean = log_clean.exp()
        del clean

    def model_kl(hs_index, v):
        with torch.no_grad(), ablate_at_block(model, hs_index, v):
            lq = model(ev_ids).logits.float().log_softmax(-1).to(lens_device)
        return float(torch.sum(p_clean * (log_clean - lq), -1).mean())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for hi in hs_list:
        # random orthonormal control (once per layer)
        d = H_ext[hi].shape[-1]
        R = torch.linalg.qr(torch.randn(d, args.k, generator=torch.Generator()
                                        .manual_seed(args.seed), device="cpu")
                            .float())[0].to(lens_device)
        for j in range(args.k):
            rows.append(dict(model=args.model, lens="random", hs=hi, j=j,
                             energy="", model_kl=round(model_kl(hi, R[:, j]), 6)))
        print(f"[{args.model}] hs{hi} random control done", flush=True)

        for name, (lens, _) in lenses.items():
            if hi not in site_of[name]:
                continue
            site = site_of[name][hi]
            t0 = time.time()
            init = svd_init(model, name, spec["lenses"][name], site, args.k,
                            lens_device)
            energies, vectors = extract_basis(lens, site, H_ext[hi], args.k,
                                              args.max_iter, init, autocast_dtype)
            for j in range(args.k):
                rows.append(dict(model=args.model, lens=name, hs=hi, j=j,
                                 energy=round(float(energies[j]), 6),
                                 model_kl=round(model_kl(hi, vectors[:, j]), 6)))
            print(f"[{args.model}] hs{hi} {name}: extracted+evaluated {args.k} dirs "
                  f"({time.time()-t0:.0f}s)", flush=True)

        with open(out_dir / "cbe.csv", "w", newline="") as f:  # checkpoint often
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # summary: causal fidelity per lens
    from scipy.stats import spearmanr
    import pandas as pd
    df = pd.read_csv(out_dir / "cbe.csv")
    print("\n== CBE causal fidelity (Spearman energy vs model KL; "
          "mean model-KL of top-8 dirs) ==")
    rand = df[df.lens == "random"].groupby("hs").model_kl.mean()
    print(f"  random-dir model KL by layer: "
          f"{ {int(h): round(v, 4) for h, v in rand.items()} }")
    for name in df.lens.unique():
        if name == "random":
            continue
        sub = df[df.lens == name]
        rhos = [spearmanr(g.energy, g.model_kl).statistic
                for _, g in sub.groupby("hs")]
        top8 = sub[sub.j < 8].model_kl.mean()
        print(f"  {name:10s} mean rho={np.mean(rhos):+.3f} "
              f"(per-layer {[round(r, 2) for r in rhos]}) top8 KL={top8:.4f}")


if __name__ == "__main__":
    main()
