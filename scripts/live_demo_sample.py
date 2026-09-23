"""
scripts/live_demo_sample.py

Live mini-run for panel: N real items through the full pipeline,
same run_pipeline() used for the 300-item run. Prints as it goes so
the panel sees real API calls happening, not a replay. Doesn't
write anything to disk — read-only against run_sample.jsonl.
"""

import json
import time
from pathlib import Path

from src.pipeline import run_pipeline
from src.schemas import BenchmarkItem

SAMPLE_PATH = Path("results/processed/run_sample.jsonl")
N_ITEMS = 5  # bump to 10 for a longer demo


def load_sample(n):
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        items = [BenchmarkItem(**json.loads(line)) for line in f]
    return items[:n]


def main():
    items = load_sample(N_ITEMS)
    print(f"Running {len(items)} real items live through all 3 critics...\n")

    n_disagree = 0
    total_cost = 0.0
    start = time.time()

    for i, item in enumerate(items, 1):
        t0 = time.time()
        result = run_pipeline(
            input_id=item.item_id,
            original_output=item.output_text,
            task_type=item.task_type,
            benchmark_item=item,
        )
        elapsed = time.time() - t0
        has_disagree = result.disagreement_matrix.has_disagreement
        n_disagree += has_disagree
        total_cost += result.total_cost_usd

        scores = {c.critic_id: c.dimension_scores for c in result.critiques}
        print(f"[{i}/{len(items)}] {item.item_id} ({elapsed:.1f}s)")
        print(f"    disagreement: {'YES' if has_disagree else 'no'}  cost: ${result.total_cost_usd:.4f}")
        print(f"    scores: {scores}\n")

    print("=" * 50)
    print(f"Done: {len(items)} items in {time.time()-start:.0f}s")
    print(f"Disagreement: {n_disagree}/{len(items)} ({100*n_disagree/len(items):.0f}%)")
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()