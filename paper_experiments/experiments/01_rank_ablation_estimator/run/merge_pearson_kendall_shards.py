#!/usr/bin/env python3
"""Merge rank-sharded Pearson/Kendall outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_key(row: dict) -> tuple:
    def maybe_int(name: str, default: int = 10**9) -> int:
        value = row.get(name, "")
        return int(value) if str(value).isdigit() else default

    family_order = {
        "lora_full": 0,
        "lora_topk": 1,
        "lora_head_is": 2,
        "tuned_topk": 3,
        "tuned_head_is": 4,
    }
    return (
        maybe_int("rank"),
        family_order.get(row.get("family", ""), 99),
        maybe_int("subset_k", 0),
        maybe_int("subset_k_tail", 0),
        maybe_int("layer", 0),
        row.get("run_name", ""),
    )


def main() -> None:
    args = parse_args()
    layer_rows: list[dict] = []
    summary_rows: list[dict] = []
    metadata = {}

    for shard_dir in sorted(args.shard_root.glob("rank_*")):
        layer_path = shard_dir / "layerwise_pearson_kendall.csv"
        summary_path = shard_dir / "rank_summary.csv"
        metadata_path = shard_dir / "metadata.json"
        if layer_path.exists():
            layer_rows.extend(read_csv(layer_path))
        if summary_path.exists():
            summary_rows.extend(read_csv(summary_path))
        if metadata_path.exists():
            metadata[shard_dir.name] = json.loads(metadata_path.read_text())

    layer_rows.sort(key=sort_key)
    summary_rows.sort(key=sort_key)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "layerwise_pearson_kendall.csv", layer_rows)
    write_csv(args.out_dir / "rank_summary.csv", summary_rows)
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"[merge] wrote {args.out_dir / 'layerwise_pearson_kendall.csv'} ({len(layer_rows)} rows)")
    print(f"[merge] wrote {args.out_dir / 'rank_summary.csv'} ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
