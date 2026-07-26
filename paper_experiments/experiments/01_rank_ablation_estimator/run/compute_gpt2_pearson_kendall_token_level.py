#!/usr/bin/env python3
"""
Recompute Pearson and Kendall correlation between GPT-2 lens checkpoints and
a full-rank baseline at the token level, rather than over the 12-dimensional
layerwise KL vector.

For each rank, layer, and sampled position we compute:
  - Pearson r between the full-vocabulary log-probability vectors of the
    candidate lens and the full-rank baseline.
  - Kendall tau between the top-K token rankings of the two lenses
    (full-vocabulary Kendall over 50k tokens per position is infeasible;
    restricting to the top-K tokens where ranking matters for
    interpretability is the right scope).

Both are then averaged across positions and reported per layer and rank.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, pearsonr
from tqdm.auto import tqdm

REPO_ROOT = Path("/path/to/project")
OMNILENS_SRC = REPO_ROOT / "omnilens" / "src"
if str(OMNILENS_SRC) not in sys.path:
    sys.path.insert(0, str(OMNILENS_SRC))

from omnilens.lenses import create_lens  # noqa: E402
from omnilens.training import HFUnembed, get_model_config  # noqa: E402


DEFAULT_EVAL_DIR = (
    REPO_ROOT / "eval_harness" / "evaluation" / "gpt2_preemptable_sweep_1000"
)
DEFAULT_OUT_DIR = DEFAULT_EVAL_DIR / "plots" / "pearson_kendall_token_level"
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
RANKS = [1, 4, 8, 16, 32, 64, 128, 256, 384]
DEFAULT_CHECKPOINT_ROOTS = [LOW_RANK_ROOT, HEAD_TAIL_FIX_ROOT, HIGH_RANK_ROOT]
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
    parser.add_argument(
        "--kendall-k",
        type=int,
        default=100,
        help=(
            "Number of top tokens over which to compute Kendall tau. "
            "Full-vocabulary Kendall (50k tokens) is infeasible per position; "
            "restricting to the top-K tokens under either lens captures the "
            "ranking structure that matters for interpretability."
        ),
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=8192,
        help=(
            "Number of token positions to sample per layer for correlation "
            "computation. Pearson over the full vocabulary is O(|V|) per "
            "position; 8192 positions gives stable estimates in reasonable time."
        ),
    )
    parser.add_argument("--tokens", type=int, default=131072)
    parser.add_argument("--max-raw-docs", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--data-name", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
            shards = sorted(
                arrow_root.glob(
                    "default-*/0.0.0/*/pile-uncopyrighted-test-*.arrow"
                )
            )
            if not shards:
                raise
            return concatenate_datasets(
                [Dataset.from_file(str(shard)) for shard in shards]
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
            f"Only built {len(samples)} samples from {len(raw)} raw docs; "
            f"requested {tokens} tokens. Increase --max-raw-docs."
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
        kwargs["bias"] = any(
            key.endswith(".bias") for key in checkpoint["lens_state_dict"]
        )
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


def lens_log_probs(lens, hidden: torch.Tensor, layer: str) -> torch.Tensor:
    """Return log-softmax over vocabulary. Output: [positions, vocab_size]."""
    if hidden.dtype != torch.float32:
        hidden = hidden.to(torch.float32)
    logits = lens(hidden, layer=layer).logits
    return F.log_softmax(logits, dim=-1)


def pearson_vectorized(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute Pearson r between corresponding rows of a and b.
    a, b: [N, D] — N positions, D vocabulary dimensions.
    Returns: [N] tensor of per-position Pearson r values.
    Computed in float64 on CPU for numerical stability.
    """
    a = a.double()
    b = b.double()
    a_centered = a - a.mean(dim=-1, keepdim=True)
    b_centered = b - b.mean(dim=-1, keepdim=True)
    num = (a_centered * b_centered).sum(dim=-1)
    denom = a_centered.norm(dim=-1) * b_centered.norm(dim=-1)
    return num / denom.clamp(min=1e-12)


def kendall_top_k(
    a: torch.Tensor, b: torch.Tensor, k: int
) -> float:
    """
    Compute mean Kendall tau between the top-k token rankings of a and b.
    a, b: [N, vocab_size] log-probability tensors (CPU, float32 or float64).
    Takes the union of top-k tokens under a and b, then computes Kendall tau
    over that candidate set. Averages across N positions.

    Scipy kendall tau is O(n log n) for n candidates; with k=100 and the
    union set bounded at 2k, this is fast enough for thousands of positions.
    """
    a = a.float()
    b = b.float()
    n = a.shape[0]
    taus = []
    top_a = a.topk(k, dim=-1).indices  # [N, k]
    top_b = b.topk(k, dim=-1).indices  # [N, k]

    for i in range(n):
        # Union of top-k indices from both lenses
        candidates = torch.cat([top_a[i], top_b[i]]).unique()
        a_scores = a[i, candidates].numpy()
        b_scores = b[i, candidates].numpy()
        tau, _ = kendalltau(a_scores, b_scores)
        if not math.isnan(tau):
            taus.append(tau)

    return float(sum(taus) / len(taus)) if taus else float("nan")


def compute_for_checkpoint(
    spec: CheckpointSpec,
    candidate_lens,
    baseline_lens,
    model,
    data: List[dict],
    max_samples: int,
    batch_size: int,
    n_positions: int,
    kendall_k: int,
    device: torch.device,
    rng: torch.Generator,
) -> Dict[str, dict]:
    """
    Collect log-prob vectors for a random sample of positions across all
    layers, then compute Pearson and Kendall statistics.

    We collect log-probs on GPU in batches, then move to CPU for the
    scipy-based Kendall computation.
    """
    num_layers = model.config.num_hidden_layers

    # Accumulators: list of [positions_so_far, vocab] tensors per layer (CPU)
    baseline_lp: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
    rank_lp: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
    collected = [0] * num_layers
    positions_needed = n_positions

    progress = tqdm(
        range(0, max_samples, batch_size),
        desc=f"{spec.run_name} — collecting logprobs",
    )

    for start in progress:
        if all(c >= positions_needed for c in collected):
            break
        stop = min(start + batch_size, max_samples)
        batch = [data[i] for i in range(start, stop)]
        input_ids = torch.tensor(
            [row["input_ids"] for row in batch], dtype=torch.long, device=device
        )
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = output.hidden_states[:-1]  # exclude final hidden state

        for layer_idx, hidden in enumerate(hidden_states):
            if collected[layer_idx] >= positions_needed:
                continue
            with torch.no_grad():
                lp_base = lens_log_probs(baseline_lens, hidden, str(layer_idx))
                lp_rank = lens_log_probs(candidate_lens, hidden, str(layer_idx))

            # Flatten [batch, seq, vocab] -> [positions, vocab]
            lp_base = lp_base.reshape(-1, lp_base.shape[-1]).cpu()
            lp_rank = lp_rank.reshape(-1, lp_rank.shape[-1]).cpu()

            # Random subsample within this batch to avoid positional bias
            n_here = lp_base.shape[0]
            remaining = positions_needed - collected[layer_idx]
            if n_here > remaining:
                idx = torch.randperm(n_here, generator=rng)[:remaining]
                lp_base = lp_base[idx]
                lp_rank = lp_rank[idx]

            baseline_lp[layer_idx].append(lp_base)
            rank_lp[layer_idx].append(lp_rank)
            collected[layer_idx] += lp_base.shape[0]

        del output, hidden_states, input_ids

    # Compute statistics per layer
    rows = {}
    for layer_idx in range(num_layers):
        base_mat = torch.cat(baseline_lp[layer_idx], dim=0)  # [N, vocab]
        rank_mat = torch.cat(rank_lp[layer_idx], dim=0)

        pearson_vals = pearson_vectorized(base_mat, rank_mat)  # [N]
        mean_pearson = pearson_vals.mean().item()

        # Kendall on a random subset of positions to keep runtime manageable
        kendall_sample = min(512, base_mat.shape[0])
        idx = torch.randperm(base_mat.shape[0], generator=rng)[:kendall_sample]
        mean_kendall = kendall_top_k(
            base_mat[idx], rank_mat[idx], k=kendall_k
        )

        rows[str(layer_idx)] = {
            "run_name": spec.run_name,
            "family": spec.family,
            "config_label": spec.config_label,
            "rank": spec.rank if spec.rank is not None else "",
            "subset_k": spec.subset_k if spec.subset_k is not None else "",
            "subset_k_tail": spec.subset_k_tail if spec.subset_k_tail is not None else "",
            "layer": layer_idx,
            "n_positions": collected[layer_idx],
            "mean_pearson": mean_pearson,
            f"mean_kendall_top{kendall_k}": mean_kendall,
        }
        print(
            f"  layer {layer_idx:2d} | "
            f"pearson={mean_pearson:.4f} | "
            f"kendall(top-{kendall_k})={mean_kendall:.4f} | "
            f"n={collected[layer_idx]}"
        )

    return rows


def write_outputs(
    out_dir: Path,
    specs: List[CheckpointSpec],
    results: Dict[str, Dict[str, dict]],
    args: argparse.Namespace,
    manifest: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    kendall_col = f"mean_kendall_top{args.kendall_k}"
    layers = sorted(
        {int(layer) for spec in specs for layer in results[spec.run_name].keys()}
    )

    with (out_dir / "metadata.json").open("w") as handle:
        json.dump(
            {"args": vars(args), "manifest": manifest},
            handle,
            indent=2,
            default=str,
        )

    layer_rows = []
    for spec in specs:
        for layer in layers:
            layer_rows.append(results[spec.run_name][str(layer)])

    with (out_dir / "layerwise_pearson_kendall.csv").open("w", newline="") as handle:
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
                "n_positions",
                "mean_pearson",
                kendall_col,
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
                "mean_pearson_all_layers",
                "mean_pearson_final_layer",
                f"mean_{kendall_col}_all_layers",
                f"mean_{kendall_col}_final_layer",
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
                    "mean_pearson_all_layers": sum(
                        r["mean_pearson"] for r in rows
                    )
                    / len(rows),
                    "mean_pearson_final_layer": rows[-1]["mean_pearson"],
                    f"mean_{kendall_col}_all_layers": sum(
                        r[kendall_col] for r in rows
                    )
                    / len(rows),
                    f"mean_{kendall_col}_final_layer": rows[-1][kendall_col],
                }
            )

    print(f"[pearson-kendall] wrote {out_dir / 'layerwise_pearson_kendall.csv'}")
    print(f"[pearson-kendall] wrote {out_dir / 'rank_summary.csv'}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = torch.Generator()
    rng.manual_seed(args.seed)
    specs = discover_checkpoints(args)
    if args.list_checkpoints:
        for spec in specs:
            print(
                f"{spec.run_name}\t{spec.family}\t{spec.config_label}\t{spec.checkpoint_path}"
            )
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
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
        "n_positions_per_layer": args.positions,
        "kendall_k": args.kendall_k,
    }

    results = {}
    with torch.no_grad():
        for spec in specs:
            print(f"\n=== {spec.run_name} ===")
            candidate_lens, _ = load_native_lens(spec.checkpoint_path, model, device)
            results[spec.run_name] = compute_for_checkpoint(
                spec=spec,
                candidate_lens=candidate_lens,
                baseline_lens=baseline_lens,
                model=model,
                data=data,
                max_samples=max_samples,
                batch_size=args.batch_size,
                n_positions=args.positions,
                kendall_k=args.kendall_k,
                device=device,
                rng=rng,
            )
            del candidate_lens
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_outputs(args.out_dir, specs, results, args, manifest)


if __name__ == "__main__":
    main()
