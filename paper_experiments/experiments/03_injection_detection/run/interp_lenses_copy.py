# Reference copy of interp_experiments/src/interp/lenses.py needed by cbe.py (which hardcodes sys.path to /path/to/project/interp/src); kept here for self-containment/reading, not imported from this location.
"""Unified lens loading + readout projection (HuggingFace-native basis).

Every toxicity / injection / agreement script used to carry its own copy of the
``create_lens`` + ``load_checkpoint_state_dict`` dance (``load_lowrank_lens``,
``load_tuned_lens``, ``load_native_lens`` …).  They only differed
in which lens type they hard-coded.  This module is the single loader:

    lens = load_lens(ckpt_dir, model, device)          # auto-detects the type
    lens = load_lens_for_method("lora_r64", model, device)

It sticks with the omnilens ``create_lens`` API (hookbox-backed), auto-detecting
``lens_type`` / ``lora_rank`` / ``lora_alpha`` from the checkpoint's own config, so
callers never re-specify them.  Also collected here: the readout helpers
(``lens_logits`` / ``project_lens`` / ``project_unembed``) and the raw
translator-matrix accessors (``lora_translator_matrix`` / ``tuned_translator_matrix``)
that the ablation direction-builders need.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

from . import config

# Method name → checkpoint directory (the canonical lens set).
CKPT_FOR_METHOD = {
    "lora_r64":     config.CKPT_LORA_R64,
    "lora_r64_tox": config.CKPT_LORA_R64_TOX,
    "tuned":        config.CKPT_TUNED,
}

# Optional per-method override of the omnilens lens type; by default each
# checkpoint is read from its own config.
LENS_TYPE_OVERRIDE = {}


# ── Checkpoint plumbing ─────────────────────────────────────────────────────

def resolve_ckpt(ckpt_dir) -> Path:
    """Latest ``lens_step_*.pt`` in ``ckpt_dir`` (by step number)."""
    ckpt_dir = Path(ckpt_dir)
    pts = sorted(ckpt_dir.glob("lens_step_*.pt"),
                 key=lambda p: int(p.stem.split("_")[-1]))
    if not pts:
        raise FileNotFoundError(f"No lens_step_*.pt in {ckpt_dir}")
    return pts[-1]


def read_site_ids(ckpt_dir) -> Optional[List[str]]:
    """Ordered ``site_id`` list from a checkpoint's ``activation_sites.csv``."""
    sites = Path(ckpt_dir) / "activation_sites.csv"
    if not sites.exists():
        return None
    with sites.open(newline="") as f:
        return [r["site_id"] for r in csv.DictReader(f)]


def module_key(site_id: str) -> str:
    """Sanitise a ``site_id`` into the key omnilens uses in its state dicts.

    e.g. ``"L05.attn_out"`` → ``"L05_x2e_attn_out"``. Matches the module-name
    escaping in omnilens' lens containers (was copy-pasted as ``_mk`` in every
    toxicity script).
    """
    return re.sub(r"[^0-9A-Za-z_]", lambda m: f"_x{ord(m.group(0)):02x}_", site_id)


# ── The one loader ──────────────────────────────────────────────────────────

def load_lens(ckpt_dir, model, device, *, lens_type: Optional[str] = None,
              dtype: Optional[torch.dtype] = None, verbose: bool = False):
    """Load any omnilens lens (``lora`` / ``tuned``) from its dir.

    The lens type, rank and alpha are read from the checkpoint's own ``config``;
    pass ``lens_type`` only to override the type recorded in the checkpoint's
    own config. Returns the lens in eval mode on ``device``; the resolved
    site ids and config are stashed on ``lens._omnilens_site_ids`` / ``._omnilens_config``.
    """
    config.add_omnilens_to_path()
    from omnilens.lenses import create_lens
    from omnilens.training import HFUnembed, get_model_config

    ckpt_dir = Path(ckpt_dir)
    ckpt_path = resolve_ckpt(ckpt_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    lt = lens_type or cfg.get("lens_type", "lora")

    site_ids = read_site_ids(ckpt_dir)
    if site_ids is None:
        site_ids = [str(i) for i in range(model.config.num_hidden_layers)]

    mc = get_model_config(model)
    kwargs = {}
    if lt == "lora":
        r = int(cfg.get("lora_rank", 64))
        kwargs["r"] = r
        kwargs["alpha"] = float(cfg.get("lora_alpha", r))
    elif lt == "tuned":
        # Match eval_harness: include a bias term iff the checkpoint has one.
        if any(k.endswith(".bias") for k in ckpt["lens_state_dict"]):
            kwargs["bias"] = True

    lens = create_lens(lt, layer_ids=site_ids, hidden_size=mc["hidden_size"],
                       unembed=HFUnembed(model), **kwargs)
    lens.load_checkpoint_state_dict(ckpt["lens_state_dict"])
    lens.to(device).eval()
    if dtype is not None:
        lens.to(dtype=dtype)

    lens._omnilens_site_ids = site_ids
    lens._omnilens_config = cfg
    if verbose:
        print(f"Loaded {lt} lens from {ckpt_path.name} ({len(site_ids)} sites)")
    return lens


def load_lens_for_method(method: str, model, device, **kwargs):
    """Load the canonical lens for a short method name (see CKPT_FOR_METHOD)."""
    if method not in CKPT_FOR_METHOD:
        raise ValueError(f"No checkpoint mapped for method {method!r}; "
                         f"known: {sorted(CKPT_FOR_METHOD)}")
    return load_lens(CKPT_FOR_METHOD[method], model, device,
                     lens_type=LENS_TYPE_OVERRIDE.get(method), **kwargs)


# ── Readout / projection helpers ────────────────────────────────────────────

def lens_logits(lens, hidden: torch.Tensor, layer: str,
                dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """``lens(hidden, layer=…).logits`` with an optional dtype cast on the input."""
    if dtype is not None and hidden.dtype != dtype:
        hidden = hidden.to(dtype)
    return lens(hidden, layer=layer).logits


def lens_log_probs(lens, hidden: torch.Tensor, layer: str) -> torch.Tensor:
    """Log-softmax over vocab of the lens readout. Casts input to float32."""
    return F.log_softmax(lens_logits(lens, hidden, layer, dtype=torch.float32), dim=-1)


def project_unembed(act: torch.Tensor, W_U: torch.Tensor) -> torch.Tensor:
    """``[…, 1, 1, d]`` last-token activation → ``[vocab]`` logit-lens logits."""
    return act[0, 0].float() @ W_U.T.float()


def project_lens(act: torch.Tensor, lens, site_id: str,
                 device: torch.device) -> torch.Tensor:
    """``[1, 1, d]`` last-token activation → ``[vocab]`` lens logits (on CPU)."""
    with torch.no_grad():
        out = lens(act.to(device), layer=site_id)
    return out.logits[0, 0].float().cpu()


# ── Raw translator matrices (for the ablation direction-builders) ───────────

def lora_translator_matrix(state_dict, site_id: str, r: int, alpha: float
                           ) -> Optional[torch.Tensor]:
    """Forward-warp matrix ``M_T = I + B·A·(alpha/r)`` for a low-rank lens site.

    Returns ``None`` if the site is absent from the state dict (caller falls back
    to the bare unembed rows).
    """
    mk = module_key(site_id)
    a_key = f"projections.{mk}.lora_A.weight"
    b_key = f"projections.{mk}.lora_B.weight"
    if a_key not in state_dict:
        return None
    A = state_dict[a_key].float()
    B = state_dict[b_key].float()
    return torch.eye(B.shape[0]) + B @ A * (alpha / r)


def tuned_translator_matrix(state_dict, site_id: str) -> Optional[torch.Tensor]:
    """Full-rank translator ``W_T`` for a tuned lens site, or ``None`` if absent."""
    w_key = f"translators.{module_key(site_id)}.weight"
    if w_key not in state_dict:
        return None
    return state_dict[w_key].float()
