"""
Dataset loading utilities.

Loads the four prompt categories from JSONL files and merges them with metadata.

Expected JSONL format per file (one JSON object per line):
  {"prompt": "the user prompt text", "source": "HEx-PHI", "id": "adv-direct-001"}

If you haven't built the datasets yet, run:
  python datasets/build_dataset.py
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal

import config


@dataclass
class TestCase:
    id: str
    prompt: str
    category: Literal[
        "adversarial_direct",
        "adversarial_indirect",
        "benign_standard",
        "benign_edgecase",
    ]
    is_adversarial: bool
    source: str = ""
    metadata: dict = field(default_factory=dict)


def _load_jsonl(path: Path, category: str, is_adversarial: bool) -> list[TestCase]:
    cases: list[TestCase] = []
    if not path.exists():
        print(f"[WARN] Dataset file not found: {path}  — skipping.")
        return cases
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            cases.append(
                TestCase(
                    id=obj.get("id", f"{category}-{len(cases)}"),
                    prompt=obj["prompt"],
                    category=category,
                    is_adversarial=is_adversarial,
                    source=obj.get("source", ""),
                    metadata=obj.get("metadata", {}),
                )
            )
    return cases


def load_all_datasets() -> list[TestCase]:
    """Load and merge all four dataset slices into a single list."""
    dataset_dir = config.DATASET_DIR

    cases: list[TestCase] = []
    cases += _load_jsonl(dataset_dir / "adversarial_direct.jsonl", "adversarial_direct", True)
    cases += _load_jsonl(dataset_dir / "adversarial_indirect.jsonl", "adversarial_indirect", True)
    cases += _load_jsonl(dataset_dir / "benign_standard.jsonl", "benign_standard", False)
    cases += _load_jsonl(dataset_dir / "benign_edgecase.jsonl", "benign_edgecase", False)

    # Deterministic shuffle to avoid time-dependent bias from loading
    # adversarial-first, benign-second.  Uses isolated RNG to avoid
    # affecting global random state.
    random.Random(42).shuffle(cases)

    print(f"[INFO] Loaded {len(cases)} total test cases (shuffled with seed=42).")
    return cases
