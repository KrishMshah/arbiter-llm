"""
src/sampling.py

Fixed-seed reproducible sample from the ground-truth-labeled datasets,
for the real 3-critic run. Equal count per dataset — keeps FActScore-
labeled (smallest, only real hallucination-span source) from getting
drowned out by MT-Bench-human (largest).
"""

import json
import random
from pathlib import Path

OUT_ROOT = Path("results/processed")

SAMPLE_SOURCES = ["truthfulqa", "summeval", "factscore_labeled", "mt_bench_human"]
SEED = 42
PER_DATASET_TARGET = 75  # Phase 2a baseline — locked, do not change

# Phase 2b: scale 300 -> 780 (195/dataset). Arena's 130-pair agreement
# sample is separate (schemas.py-blocked, different sampler entirely).
EXPANSION_PER_DATASET_TARGET = 195
ADDITIONAL_PER_DATASET = EXPANSION_PER_DATASET_TARGET - PER_DATASET_TARGET  # 120


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _write_jsonl(items: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def build_sample() -> list[dict]:
    """Phase 2a baseline — unchanged, kept for reproducibility."""
    rng = random.Random(SEED)
    sample = []

    for name in SAMPLE_SOURCES:
        items = _load_jsonl(OUT_ROOT / f"{name}.jsonl")
        n = min(PER_DATASET_TARGET, len(items))
        sample.extend(rng.sample(items, n))

    rng.shuffle(sample)
    return sample


def build_expansion_sample(existing_sample_path: Path) -> list[dict]:
    """
    120/dataset new items on top of the existing 300. Excludes anything
    already in existing_sample_path by item_id, so the already-processed
    300 never get re-run through the paid critics.
    """
    existing_ids = {item["item_id"] for item in _load_jsonl(existing_sample_path)}

    rng = random.Random(SEED)  # fresh stream over the post-exclusion pool — deterministic, independent of build_sample()
    expansion = []

    for name in SAMPLE_SOURCES:
        items = _load_jsonl(OUT_ROOT / f"{name}.jsonl")
        eligible = [item for item in items if item["item_id"] not in existing_ids]

        n = min(ADDITIONAL_PER_DATASET, len(eligible))
        if n < ADDITIONAL_PER_DATASET:
            print(f"WARNING: {name} has only {len(eligible)} eligible items left, wanted {ADDITIONAL_PER_DATASET}")
        expansion.extend(rng.sample(eligible, n))

    rng.shuffle(expansion)
    return expansion


if __name__ == "__main__":
    existing_path = OUT_ROOT / "run_sample.jsonl"
    expansion = build_expansion_sample(existing_path)

    out_path = OUT_ROOT / "run_sample_expansion.jsonl"
    _write_jsonl(expansion, out_path)

    meta = {
        "seed": SEED,
        "additional_per_dataset_target": ADDITIONAL_PER_DATASET,
        "expansion_total_items": len(expansion),
        "cumulative_total_items": len(expansion) + PER_DATASET_TARGET * len(SAMPLE_SOURCES),
        "sources": SAMPLE_SOURCES,
        "base_sample": str(existing_path),
    }
    with open(OUT_ROOT / "run_sample_expansion_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Sampled {len(expansion)} new items -> {out_path}")
    print(json.dumps(meta, indent=2))