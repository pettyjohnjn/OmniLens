"""Cross-lens DART comparison over all toxic_heads_*.csv files per model.

Per lens: concentration stats (total toxic count, top head + share, top-5 share,
late/early layer skew). Pairwise between lenses: Pearson / Spearman / Kendall of
the flattened per-head counts, and top-25% / top-50% set overlap of head ranks.

Output: dart_crosslens_summary.csv + dart_crosslens_pairs.csv + printed tables.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
MODELS = {
    "GPT-2":       ROOT / "results_gpt2/dart",
    "LLaMA-3-8B":  ROOT / "results_llama8b/dart",
    "LLaMA-3-70B": ROOT / "results_llama70b/dart",
}


def load_counts(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    return df[[c for c in df.columns if c.startswith("head_")]].values.astype(float)


def concentration(c: np.ndarray) -> dict:
    tot = c.sum() or 1.0
    flat = c.flatten()
    order = np.argsort(flat)[::-1]
    l, h = np.unravel_index(order[0], c.shape)
    nL = c.shape[0]
    early, late = c[: nL // 2].sum(), c[nL // 2:].sum()
    return dict(total=int(c.sum()),
                top_head=f"L{l:02d}.H{h:02d}", top_share=100 * flat[order[0]] / tot,
                top5_share=100 * flat[order[:5]].sum() / tot,
                late_early=(late / early) if early > 0 else np.inf)


def top_overlap(a: np.ndarray, b: np.ndarray, frac: float) -> float:
    k = max(1, int(len(a) * frac))
    sa = set(np.argsort(a)[::-1][:k])
    sb = set(np.argsort(b)[::-1][:k])
    return len(sa & sb) / k


def main():
    conc_rows, pair_rows = [], []
    for mname, d in MODELS.items():
        if not d.exists():
            continue
        lenses = {p.stem.replace("toxic_heads_", ""): load_counts(p)
                  for p in sorted(d.glob("toxic_heads_*.csv"))}
        for tag, c in lenses.items():
            conc_rows.append(dict(model=mname, lens=tag, **concentration(c)))
        for (ta, ca), (tb, cb) in combinations(lenses.items(), 2):
            fa, fb = ca.flatten(), cb.flatten()
            pair_rows.append(dict(
                model=mname, pair=f"{ta} vs {tb}",
                pearson=stats.pearsonr(fa, fb)[0],
                spearman=stats.spearmanr(fa, fb)[0],
                kendall=stats.kendalltau(fa, fb)[0],
                top25=top_overlap(fa, fb, 0.25),
                top50=top_overlap(fa, fb, 0.50),
                same_top5=len(set(np.argsort(fa)[::-1][:5]) & set(np.argsort(fb)[::-1][:5])),
            ))

    conc = pd.DataFrame(conc_rows)
    pairs = pd.DataFrame(pair_rows)
    conc.to_csv(ROOT / "dart_crosslens_summary.csv", index=False)
    pairs.to_csv(ROOT / "dart_crosslens_pairs.csv", index=False)
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print("=== concentration ===")
    print(conc.to_string(index=False, float_format=lambda x: f"{x:8.3g}"))
    print("\n=== pairwise agreement ===")
    print(pairs.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\nwrote dart_crosslens_summary.csv, dart_crosslens_pairs.csv")


if __name__ == "__main__":
    main()
