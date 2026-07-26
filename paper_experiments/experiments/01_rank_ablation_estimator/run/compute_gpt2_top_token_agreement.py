#!/usr/bin/env python3
"""Measure top-token agreement between GPT-2 lens checkpoints and a baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path("/path/to/project")
OMNILENS_SRC = REPO_ROOT / "omnilens" / "src"
if str(OMNILENS_SRC) not in sys.path:
    sys.path.insert(0, str(OMNILENS_SRC))

from omnilens.lenses import create_lens  # noqa: E402
from omnilens.training import HFUnembed, get_model_config  # noqa: E402


DEFAULT_EVAL_DIR = REPO_ROOT / "eval_harness" / "evaluation" / "gpt2_preemptable_sweep_1000"
DEFAULT_OUT_DIR = DEFAULT_EVAL_DIR / "plots" / "top_token_agreement"
DEFAULT_BASELINE = (
    REPO_ROOT
    / "omnilens"
    / "src"
    / "checkpoints"
    / "gpt2"
    / "gpt2_preemptable_sweep_1000"
    / "tuned_kl_baseline"
    / "lens_step_1000.pt"
)
LOW_RANK_ROOT = (
    REPO_ROOT
    / "omnilens"
    / "src"
    / "checkpoints"
    / "gpt2_debug_lowrank_sweep_1000"
)
HEAD_TAIL_FIX_ROOT = (
    REPO_ROOT
    / "omnilens"
    / "src"
    / "checkpoints"
    / "gpt2"
    / "gpt2_preemptable_head_is_fix_1000"
)
HIGH_RANK_ROOT = (
    REPO_ROOT
    / "omnilens"
    / "src"
    / "checkpoints"
    / "gpt2"
    / "gpt2_preemptable_sweep_1000"
)
DEFAULT_CHECKPOINT_ROOTS = [LOW_RANK_ROOT, HEAD_TAIL_FIX_ROOT, HIGH_RANK_ROOT]
RANKS = [1, 4, 8, 16, 32, 64, 128, 256, 384]
DEFAULT_FAMILIES = ["lora_full", "lora_topk", "lora_head_is", "tuned_topk", "tuned_head_is"]


@dataclass(frozen=True)
class CheckpointSpec:
    run_name: str
    checkpoint_path: Path
    family: str
    rank: Optional[int] = None
    subset_k: Optional[int] = None
    subset_k_tail: Optional[int] = None

    @property
    def config_label(self) -> str:
        if self.family == "lora_full":
            return f"LoRA full-KL r{self.rank}"
        if self.family == "lora_topk":
            return f"LoRA top-k r{self.rank} k{self.subset_k}"
        if self.family == "lora_head_is":
            return f"LoRA head+IS r{self.rank} {self.subset_k}+{self.subset_k_tail}"
        if self.family == "tuned_topk":
            return f"Tuned top-k k{self.subset_k}"
        if self.family == "tuned_head_is":
            return f"Tuned head+IS {self.subset_k}+{self.subset_k_tail}"
        return self.run_name

    @property
    def sort_key(self) -> tuple:
        family_order = {
            "lora_full": 0,
            "lora_topk": 1,
            "lora_head_is": 2,
            "tuned_topk": 3,
            "tuned_head_is": 4,
        }
        return (
            self.rank if self.rank is not None else 10**9,
            family_order.get(self.family, 99),
            self.subset_k or 0,
            self.subset_k_tail or 0,
            self.run_name,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument(
        "--checkpoint-roots",
        type=Path,
        nargs="+",
        default=DEFAULT_CHECKPOINT_ROOTS,
        help="Roots containing run_name/lens_step_<step>.pt checkpoints.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=DEFAULT_FAMILIES,
        choices=["lora_full", "lora_topk", "lora_head_is", "tuned_topk", "tuned_head_is"],
        help="Checkpoint families to evaluate.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=None,
        help="Optional rank filter for LoRA checkpoints. Omit to include all discovered ranks.",
    )
    parser.add_argument(
        "--run-regex",
        default=None,
        help="Optional regex filter applied to run names after family/rank filtering.",
    )
    parser.add_argument(
        "--exclude-run-regex",
        default=None,
        help="Optional regex exclusion applied to run names.",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List matched checkpoints and exit before loading the model.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=131072)
    parser.add_argument(
        "--max-raw-docs",
        type=int,
        default=8192,
        help="Maximum raw Pile documents to tokenize before taking the token budget.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--data-name", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load model/tokenizer from the local Hugging Face cache.",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str):
    return {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def default_rank_checkpoint(rank: int, step: int) -> Path:
    root = LOW_RANK_ROOT if rank <= 32 else HIGH_RANK_ROOT
    return root / f"lora_kl_r{rank}" / f"lens_step_{step}.pt"


def spec_from_run_dir(run_dir: Path, step: int) -> Optional[CheckpointSpec]:
    if step < 0:
        checkpoints = sorted(
            run_dir.glob("lens_step_*.pt"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        checkpoint_path = checkpoints[-1] if checkpoints else run_dir / "lens_step_-1.pt"
    else:
        checkpoint_path = run_dir / f"lens_step_{step}.pt"
    if not checkpoint_path.exists():
        return None
    name = run_dir.name
    patterns = [
        ("lora_full", r"^lora_kl_r(?P<rank>\d+)$"),
        ("lora_topk", r"^lora_subset_topk_r(?P<rank>\d+)_k(?P<k>\d+)$"),
        ("lora_head_is", r"^lora_subset_head_is_r(?P<rank>\d+)_k(?P<k>\d+)_tail(?P<tail>\d+)$"),
        ("lora_head_is", r"^.*lora_subset_is_gpt2_expanded_r(?P<rank>\d+)_k(?P<k>\d+)_tail(?P<tail>\d+).*$"),
        ("lora_head_is", r"^.*lora_subset_is_r(?P<rank>\d+)_k(?P<k>\d+)_tail(?P<tail>\d+).*$"),
        ("lora_topk", r"^.*lora_r(?P<rank>\d+)_subsetkl_topk(?P<k>\d+).*$"),
        ("lora_head_is", r"^.*lora_r(?P<rank>\d+)_subsetkl_head_is(?P<k>\d+)_tail(?P<tail>\d+).*$"),
        ("tuned_topk", r"^tuned_subset_topk_k(?P<k>\d+)$"),
        ("tuned_head_is", r"^tuned_subset_head_is_k(?P<k>\d+)_tail(?P<tail>\d+)$"),
    ]
    for family, pattern in patterns:
        match = re.fullmatch(pattern, name)
        if match is None:
            continue
        groups = match.groupdict()
        return CheckpointSpec(
            run_name=name,
            checkpoint_path=checkpoint_path,
            family=family,
            rank=int(groups["rank"]) if groups.get("rank") else None,
            subset_k=int(groups["k"]) if groups.get("k") else None,
            subset_k_tail=int(groups["tail"]) if groups.get("tail") else None,
        )
    return None


def discover_checkpoints(args: argparse.Namespace) -> List[CheckpointSpec]:
    specs_by_name: Dict[str, CheckpointSpec] = {}
    families = set(args.families)
    ranks = set(args.ranks) if args.ranks else None
    include_re = re.compile(args.run_regex) if args.run_regex else None
    exclude_re = re.compile(args.exclude_run_regex) if args.exclude_run_regex else None

    for root in args.checkpoint_roots:
        if not root.exists():
            continue
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            spec = spec_from_run_dir(run_dir, args.step)
            if spec is None or spec.family not in families:
                continue
            if ranks is not None and spec.rank is not None and spec.rank not in ranks:
                continue
            if include_re is not None and include_re.search(spec.run_name) is None:
                continue
            if exclude_re is not None and exclude_re.search(spec.run_name):
                continue
            specs_by_name.setdefault(spec.run_name, spec)

    specs = sorted(specs_by_name.values(), key=lambda spec: spec.sort_key)
    if not specs:
        raise FileNotFoundError(
            "No checkpoints matched the requested roots/families/ranks/regex filters."
        )
    return specs


def load_raw_dataset(name: str):
    from datasets import Dataset, concatenate_datasets, load_dataset

    if name.endswith(".jsonl"):
        return Dataset.from_json(name)
    elif name == "test":
        try:
            return load_dataset(
                "monology/pile-uncopyrighted",
                data_files={"test": "test.jsonl.zst"},
                split="test",
                cache_dir="./mycachedir/",
            )
        except ValueError:
            arrow_root = Path("mycachedir/monology___pile-uncopyrighted").resolve()
            shards = sorted(arrow_root.glob("default-*/0.0.0/*/pile-uncopyrighted-test-*.arrow"))
            if not shards:
                raise
            return concatenate_datasets([Dataset.from_file(str(shard)) for shard in shards])
    elif name == "val":
        return load_dataset(
            "monology/pile-uncopyrighted",
            data_files={"validation": "val.jsonl.zst"},
            split="validation",
            cache_dir="./mycachedir/",
        )
    else:
        return load_dataset(name, split="validation")


def load_tokenized_samples(
    name: str,
    tokenizer,
    max_seq_len: int,
    seed: int,
    tokens: int,
    max_raw_docs: int,
) -> List[dict]:
    raw = load_raw_dataset(name)

    raw = raw.shuffle(seed=seed)
    if max_raw_docs > 0 and len(raw) > max_raw_docs:
        raw = raw.select(range(max_raw_docs))

    needed = max(1, math.ceil(tokens / max_seq_len))
    sep = tokenizer.eos_token or "<|endoftext|>"
    sep_ids = tokenizer.encode(sep, add_special_tokens=False)
    buffer: List[int] = []
    samples: List[dict] = []

    for row in raw:
        text = row.get("text")
        if not text:
            continue
        buffer.extend(sep_ids)
        buffer.extend(
            tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
                verbose=False,
            )
        )
        while len(buffer) >= max_seq_len and len(samples) < needed:
            samples.append({"input_ids": buffer[:max_seq_len]})
            del buffer[:max_seq_len]
        if len(samples) >= needed:
            break

    if len(samples) < needed:
        raise ValueError(
            f"Only built {len(samples)} samples ({len(samples) * max_seq_len} tokens) "
            f"from {len(raw)} raw docs; requested {tokens} tokens. Increase --max-raw-docs."
        )
    return samples


def infer_site_ids(checkpoint_path: Path, model) -> List[str]:
    activation_sites = checkpoint_path.parent / "activation_sites.csv"
    if activation_sites.exists():
        with activation_sites.open(newline="") as handle:
            return [row["site_id"] for row in csv.DictReader(handle)]
    return [str(idx) for idx in range(model.config.num_hidden_layers)]


def load_native_lens(checkpoint_path: Path, model, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    lens_type = config["lens_type"]
    site_ids = infer_site_ids(checkpoint_path, model)
    model_cfg = get_model_config(model)
    unembed = HFUnembed(model).clone_to_device(device)

    kwargs = {}
    if lens_type == "lora":
        kwargs["r"] = int(config["lora_rank"])
        kwargs["alpha"] = float(config.get("lora_alpha", 1.0))
    elif lens_type == "tuned":
        kwargs["bias"] = any(key.endswith(".bias") for key in checkpoint["lens_state_dict"])
    else:
        raise ValueError(f"Unsupported lens type: {lens_type!r}")

    lens = create_lens(
        lens_type,
        layer_ids=site_ids,
        hidden_size=model_cfg["hidden_size"],
        unembed=unembed,
        **kwargs,
    )
    lens.load_checkpoint_state_dict(checkpoint["lens_state_dict"])
    lens.to(device=device, dtype=torch.float32)
    lens.eval()
    return lens, site_ids


def batch_input_ids(batch: List[dict], device: torch.device) -> torch.Tensor:
    return torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)


def iter_batches(data, batch_size: int, max_samples: int) -> Iterable[List[dict]]:
    for start in range(0, max_samples, batch_size):
        stop = min(start + batch_size, max_samples)
        yield [data[idx] for idx in range(start, stop)]


def topk_overlap_fraction(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    # candidate/reference: [positions, k]
    matches = candidate.unsqueeze(-1).eq(reference.unsqueeze(-2)).any(dim=-1)
    return matches.float().sum(dim=-1) / candidate.shape[-1]


def lens_param_dtype(lens) -> torch.dtype:
    for param in lens.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def lens_logits(lens, hidden: torch.Tensor, layer: str, dtype: torch.dtype) -> torch.Tensor:
    if hidden.dtype != dtype:
        hidden = hidden.to(dtype)
    return lens(hidden, layer=layer).logits


def compute_for_checkpoint(
    spec: CheckpointSpec,
    candidate_lens,
    baseline_lens,
    model,
    data,
    max_samples: int,
    batch_size: int,
    top_k: int,
    device: torch.device,
) -> Dict[str, dict]:
    num_layers = model.config.num_hidden_layers
    baseline_dtype = lens_param_dtype(baseline_lens)
    candidate_dtype = lens_param_dtype(candidate_lens)
    top1_correct = torch.zeros(num_layers, dtype=torch.float64, device=device)
    topk_overlap = torch.zeros(num_layers, dtype=torch.float64, device=device)
    counts = torch.zeros(num_layers, dtype=torch.float64, device=device)

    progress = tqdm(
        iter_batches(data, batch_size, max_samples),
        total=math.ceil(max_samples / batch_size),
        desc=spec.run_name,
    )
    for batch in progress:
        input_ids = batch_input_ids(batch, device)
        output = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hidden_states = output.hidden_states[:-1]

        for layer_idx, hidden in enumerate(hidden_states):
            layer = str(layer_idx)
            baseline_logits = lens_logits(baseline_lens, hidden, layer, baseline_dtype)
            candidate_logits = lens_logits(candidate_lens, hidden, layer, candidate_dtype)

            baseline_top = baseline_logits.topk(top_k, dim=-1).indices.reshape(-1, top_k)
            candidate_top = candidate_logits.topk(top_k, dim=-1).indices.reshape(-1, top_k)
            n_pos = baseline_top.shape[0]

            top1_correct[layer_idx] += candidate_top[:, 0].eq(baseline_top[:, 0]).sum()
            topk_overlap[layer_idx] += topk_overlap_fraction(candidate_top, baseline_top).sum()
            counts[layer_idx] += n_pos

            del baseline_logits, candidate_logits, baseline_top, candidate_top

        del output, hidden_states, input_ids

    rows = {}
    for layer_idx in range(num_layers):
        denom = counts[layer_idx].item()
        rows[str(layer_idx)] = {
            "run_name": spec.run_name,
            "family": spec.family,
            "config_label": spec.config_label,
            "rank": spec.rank if spec.rank is not None else "",
            "subset_k": spec.subset_k if spec.subset_k is not None else "",
            "subset_k_tail": spec.subset_k_tail if spec.subset_k_tail is not None else "",
            "layer": layer_idx,
            "positions": int(denom),
            "top1_agreement": (top1_correct[layer_idx] / counts[layer_idx]).item(),
            f"top{top_k}_overlap": (topk_overlap[layer_idx] / counts[layer_idx]).item(),
        }
    return rows


def write_outputs(out_dir: Path, specs: List[CheckpointSpec], results: Dict[str, Dict[str, dict]], args: argparse.Namespace, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    topk_name = f"top{args.top_k}_overlap"
    layers = sorted(
        {int(layer) for spec in specs for layer in results[spec.run_name].keys()}
    )

    with (out_dir / "metadata.json").open("w") as handle:
        json.dump({"args": vars(args), "manifest": manifest}, handle, indent=2, default=str)

    layer_rows = []
    for spec in specs:
        for layer in layers:
            layer_rows.append(results[spec.run_name][str(layer)])

    with (out_dir / "layerwise_top_token_agreement.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "family",
                "config_label",
                "rank",
                "subset_k",
                "subset_k_tail",
                "layer",
                "positions",
                "top1_agreement",
                topk_name,
            ],
        )
        writer.writeheader()
        writer.writerows(layer_rows)

    with (out_dir / "rank_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "family",
                "config_label",
                "rank",
                "subset_k",
                "subset_k_tail",
                "mean_top1_agreement",
                f"mean_{topk_name}",
                "final_top1_agreement",
                f"final_{topk_name}",
            ],
        )
        writer.writeheader()
        for spec in specs:
            rows = [results[spec.run_name][str(layer)] for layer in layers]
            writer.writerow(
                {
                    "run_name": spec.run_name,
                    "family": spec.family,
                    "config_label": spec.config_label,
                    "rank": spec.rank if spec.rank is not None else "",
                    "subset_k": spec.subset_k if spec.subset_k is not None else "",
                    "subset_k_tail": spec.subset_k_tail if spec.subset_k_tail is not None else "",
                    "mean_top1_agreement": sum(row["top1_agreement"] for row in rows) / len(rows),
                    f"mean_{topk_name}": sum(row[topk_name] for row in rows) / len(rows),
                    "final_top1_agreement": rows[-1]["top1_agreement"],
                    f"final_{topk_name}": rows[-1][topk_name],
                }
            )


def plot_outputs(out_dir: Path, args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    topk_name = f"top{args.top_k}_overlap"
    rows = []
    with (out_dir / "layerwise_top_token_agreement.csv").open() as handle:
        for row in csv.DictReader(handle):
            rows.append(row)

    labels = []
    for row in rows:
        if row["layer"] == "0":
            labels.append(row["config_label"])
    layers = sorted({int(row["layer"]) for row in rows})
    top1 = np.zeros((len(layers), len(labels)))
    topk = np.zeros_like(top1)
    by_key = {(int(row["layer"]), row["run_name"]): row for row in rows}
    run_names = [row["run_name"] for row in rows if row["layer"] == "0"]
    for y, layer in enumerate(layers):
        for x, run_name in enumerate(run_names):
            row = by_key[(layer, run_name)]
            top1[y, x] = float(row["top1_agreement"])
            topk[y, x] = float(row[topk_name])

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    width = max(7.2, min(22.0, 0.34 * len(labels) + 3.2))
    fig, axes = plt.subplots(1, 2, figsize=(width, 3.3), constrained_layout=True)
    panels = [(axes[0], top1, "Top-1 agreement"), (axes[1], topk, f"Top-{args.top_k} overlap")]
    for ax, matrix, title in panels:
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="mako" if False else "viridis", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=55, ha="right")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(layer) for layer in layers])
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("Agreement")

    fig.savefig(out_dir / "top_token_agreement_heatmap.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "top_token_agreement_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for matrix, stem, title in [
        (top1, "top1_agreement_heatmap", "Top-1 agreement"),
        (topk, f"top{args.top_k}_overlap_heatmap", f"Top-{args.top_k} overlap"),
    ]:
        fig, ax = plt.subplots(figsize=(max(3.65, min(14.0, 0.34 * len(labels) + 1.8)), 3.2), constrained_layout=True)
        im = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=55, ha="right")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(layer) for layer in layers])
        cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.035)
        cbar.set_label("Agreement")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    specs = discover_checkpoints(args)
    if args.list_checkpoints:
        for spec in specs:
            print(
                f"{spec.run_name}\t{spec.family}\t{spec.config_label}\t{spec.checkpoint_path}"
            )
        return

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_from_name(args.dtype),
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    data = load_tokenized_samples(
        args.data_name,
        tokenizer,
        args.max_seq_len,
        args.seed,
        args.tokens,
        args.max_raw_docs,
    )
    max_samples = min(len(data), max(1, math.ceil(args.tokens / args.max_seq_len)))

    baseline_lens, _ = load_native_lens(args.baseline_checkpoint, model, device)
    manifest = {
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "checkpoints": [asdict(spec) for spec in specs],
        "max_samples": max_samples,
        "positions_per_layer": max_samples * args.max_seq_len,
    }

    results = {}
    with torch.no_grad():
        for spec in specs:
            candidate_lens, _ = load_native_lens(spec.checkpoint_path, model, device)
            results[spec.run_name] = compute_for_checkpoint(
                spec=spec,
                candidate_lens=candidate_lens,
                baseline_lens=baseline_lens,
                model=model,
                data=data,
                max_samples=max_samples,
                batch_size=args.batch_size,
                top_k=args.top_k,
                device=device,
            )
            del candidate_lens
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_outputs(args.out_dir, specs, results, args, manifest)
    plot_ok = False
    if not args.no_plot:
        try:
            plot_outputs(args.out_dir, args)
            plot_ok = True
        except Exception as exc:
            print(f"[top-token-agreement] plot generation failed: {exc}", file=sys.stderr)

    print(f"[top-token-agreement] wrote {args.out_dir / 'layerwise_top_token_agreement.csv'}")
    print(f"[top-token-agreement] wrote {args.out_dir / 'rank_summary.csv'}")
    if plot_ok:
        print(f"[top-token-agreement] wrote {args.out_dir / 'top_token_agreement_heatmap.pdf'}")


if __name__ == "__main__":
    main()
