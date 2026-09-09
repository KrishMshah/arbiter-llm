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
PER_DATASET_TARGET = 75  # ~300 total across 4 sources


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_sample() -> list[dict]:
    rng = random.Random(SEED)
    sample = []

    for name in SAMPLE_SOURCES:
        items = _load_jsonl(OUT_ROOT / f"{name}.jsonl")
        n = min(PER_DATASET_TARGET, len(items))
        sample.extend(rng.sample(items, n))

    rng.shuffle(sample)
    return sample


if __name__ == "__main__":
    sample = build_sample()

    out_path = OUT_ROOT / "run_sample.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in sample:
            f.write(json.dumps(item) + "\n")

    meta = {
        "seed": SEED,
        "per_dataset_target": PER_DATASET_TARGET,
        "total_items": len(sample),
        "sources": SAMPLE_SOURCES,
    }
    with open(OUT_ROOT / "run_sample_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Sampled {len(sample)} items -> {out_path}")
    print(json.dumps(meta, indent=2))