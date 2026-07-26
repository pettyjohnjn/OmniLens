"""Shared toxicity-experiment helpers (dataset + dictionary loading).

Only the pieces that were *byte-for-byte identical* across the toxicity scripts
live here. The ``measure_toxicity`` / ``measure_perplexity`` routines are
intentionally NOT shared: ``run_hookset_ablation`` and ``toxicity_ablation``
use deliberately different generation/perplexity settings, and unifying them would
silently change already-reported numbers.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

TOXIC_DICT = Path(__file__).parent / "toxic_dictionary.txt"

# The 43-token toxic "memory" used to form ablation directions (identical in
# run_hookset_ablation and toxicity_ablation).
DEFAULT_MEMORY_TEXT = (
    "sucker fucking fuck piss goddamn shit shitty whore suck filthy liar "
    "pathetic fools cheating bitch booze Fuck idiots poop sneaky whine asshole "
    "sucks crap slut fucked bullshit FUCK torment lazy dope damned screw miser "
    "damn shameless stupid lousy idiot drunken vomit"
)


def load_toxic_words(path: Path = TOXIC_DICT) -> set:
    """Lower-cased toxic dictionary (one word per line)."""
    return {w.strip().lower() for w in Path(path).read_text().splitlines() if w.strip()}


def load_toxic_prompts(n: int, dataset_name: str = "OxAISH-AL-LLM/wiki_toxic") -> List[str]:
    """First ``n`` label==1 (toxic) comments from the wiki_toxic train split."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="train")
    prompts: List[str] = []
    for row in dataset:
        if row.get("label") != 1:
            continue
        text = row.get("comment_text", "")
        if isinstance(text, str) and text.strip():
            prompts.append(text.strip())
        if len(prompts) >= n:
            break
    return prompts
