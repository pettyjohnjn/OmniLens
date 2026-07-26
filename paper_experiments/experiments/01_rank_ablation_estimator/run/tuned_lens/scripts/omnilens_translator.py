import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Pattern

import torch


DEFAULT_CONFIG = {
    "base_model_name_or_path": "gpt2",
    "base_model_revision": None,
    "d_model": 768,
    "num_hidden_layers": 12,
    "bias": True,
    "unembed_hash": None,
}

LORA_KEY_RE = re.compile(r"^projections\.(.+)\.(lora_A\.weight|lora_B\.weight|bias)$")
TUNED_KEY_RE = re.compile(r"^translators\.(.+)\.(weight|bias)$")


def module_safe_layer_key(layer_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", lambda m: f"_x{ord(m.group(0)):02x}_", layer_id)


def read_activation_sites(src_ckpt: Path) -> List[Dict[str, str]]:
    path = src_ckpt.parent / "activation_sites.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def infer_hidden_size(state: Dict[str, torch.Tensor]) -> int:
    for key, value in state.items():
        if key.endswith(".lora_A.weight"):
            return int(value.shape[1])
        if key.endswith(".weight") and value.ndim == 2:
            return int(value.shape[1])
    raise ValueError("Unable to infer hidden size from checkpoint state.")


def infer_lens_config(
    ckpt: dict,
    *,
    state: Dict[str, torch.Tensor],
    activation_sites: List[Dict[str, str]],
) -> dict:
    train_cfg = ckpt.get("config", {})
    config = dict(DEFAULT_CONFIG)
    config["base_model_name_or_path"] = train_cfg.get(
        "model_name", DEFAULT_CONFIG["base_model_name_or_path"]
    )
    config["d_model"] = infer_hidden_size(state)
    if activation_sites:
        config["num_hidden_layers"] = len(activation_sites)
    config["bias"] = any(key.endswith(".bias") for key in state)
    return config


def ordered_module_keys(
    state: Dict[str, torch.Tensor],
    *,
    activation_sites: List[Dict[str, str]],
    pattern: Pattern,
) -> List[str]:
    if activation_sites:
        return [module_safe_layer_key(row["site_id"]) for row in activation_sites]

    discovered = set()
    for key in state:
        match = pattern.match(key)
        if match is not None:
            discovered.add(match.group(1))
    if not discovered:
        raise ValueError("No per-site translator keys found in checkpoint.")
    return sorted(discovered)


def convert_tuned_state(
    state: Dict[str, torch.Tensor],
    *,
    activation_sites: List[Dict[str, str]],
) -> Dict[str, torch.Tensor]:
    module_keys = ordered_module_keys(
        state,
        activation_sites=activation_sites,
        pattern=TUNED_KEY_RE,
    )
    converted = {}
    for idx, module_key in enumerate(module_keys):
        weight_key = f"translators.{module_key}.weight"
        bias_key = f"translators.{module_key}.bias"

        if weight_key not in state:
            raise ValueError(f"Missing tuned translator weights for site key {module_key!r}.")

        converted[f"{idx}.weight"] = state[weight_key]
        if bias_key in state:
            converted[f"{idx}.bias"] = state[bias_key]

    return converted


def convert_lora_state(
    ckpt: dict,
    state: Dict[str, torch.Tensor],
    *,
    activation_sites: List[Dict[str, str]],
) -> Dict[str, torch.Tensor]:
    train_cfg = ckpt.get("config", {})
    rank = train_cfg.get("lora_rank")
    alpha = train_cfg.get("lora_alpha", 1.0)
    if not rank:
        raise ValueError("LoRA checkpoint is missing config.lora_rank.")

    scaling = alpha / rank
    module_keys = ordered_module_keys(
        state,
        activation_sites=activation_sites,
        pattern=LORA_KEY_RE,
    )

    converted = {}
    for idx, module_key in enumerate(module_keys):
        a_key = f"projections.{module_key}.lora_A.weight"
        b_key = f"projections.{module_key}.lora_B.weight"
        bias_key = f"projections.{module_key}.bias"

        if a_key not in state or b_key not in state:
            raise ValueError(f"Site key {module_key!r} is missing LoRA matrices.")

        a = state[a_key]
        b = state[b_key]
        converted[f"{idx}.weight"] = scaling * (b @ a)

        if bias_key in state:
            converted[f"{idx}.bias"] = state[bias_key]

    return converted


def convert_checkpoint(src_ckpt: Path, out_dir: Path) -> None:
    ckpt = torch.load(src_ckpt, map_location="cpu")
    state = ckpt["lens_state_dict"]
    activation_sites = read_activation_sites(src_ckpt)

    if any(key.startswith("translators.") for key in state):
        converted = convert_tuned_state(state, activation_sites=activation_sites)
        source_type = "tuned"
    elif any(key.startswith("projections.") for key in state):
        converted = convert_lora_state(
            ckpt,
            state,
            activation_sites=activation_sites,
        )
        source_type = "lora"
    else:
        raise ValueError(f"Unsupported checkpoint format: {src_ckpt}")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(converted, out_dir / "params.pt")

    with (out_dir / "config.json").open("w") as f:
        json.dump(
            infer_lens_config(
                ckpt,
                state=state,
                activation_sites=activation_sites,
            ),
            f,
        )

    activation_sites_path = src_ckpt.parent / "activation_sites.csv"
    if activation_sites_path.exists():
        shutil.copy2(activation_sites_path, out_dir / "activation_sites.csv")

    print(f"[ok] {src_ckpt} -> {out_dir} ({source_type})")


def find_checkpoints(src: Path) -> List[Path]:
    if src.is_file():
        return [src]

    direct = sorted(src.glob("lens_step_*.pt"))
    if direct:
        return direct

    return sorted(src.glob("*/lens_step_*.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OmniLens or tuned-lens-style checkpoints into tuned-lens artifacts."
    )
    parser.add_argument(
        "src",
        type=Path,
        help="Checkpoint file, run directory, or a directory containing run subdirectories.",
    )
    parser.add_argument(
        "out_root",
        type=Path,
        help="Output directory. For directory input, one child directory is created per checkpoint.",
    )
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.src)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {args.src}")

    multi_input = args.src.is_dir()
    for ckpt_path in checkpoints:
        run_name = ckpt_path.parent.name
        step_name = ckpt_path.stem
        out_dir = args.out_root / f"{run_name}__{step_name}" if multi_input else args.out_root
        convert_checkpoint(ckpt_path, out_dir)


if __name__ == "__main__":
    main()
