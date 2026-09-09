"""
src/run_real_sample.py

Runs run_sample.jsonl through the full pipeline, all 3 critics real.
Writes incrementally so a crash mid-run doesn't lose completed items.
"""

import json
import time
from pathlib import Path

from src.config import MOCK_CRITIC_A, MOCK_CRITIC_B, MOCK_CRITIC_C, OLLAMA_MODEL
from src.pipeline import run_pipeline
from src.schemas import BenchmarkItem

OUT_ROOT = Path("results/processed")
SAMPLE_PATH = OUT_ROOT / "run_sample.jsonl"
RESULTS_PATH = OUT_ROOT / "run_results.jsonl"
META_PATH = OUT_ROOT / "run_metadata.json"


def load_sample() -> list[BenchmarkItem]:
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        return [BenchmarkItem(**json.loads(line)) for line in f]


def main():
    items = load_sample()
    print(f"Running {len(items)} items through the full pipeline (all 3 critics real)...")

    n_ok, failures = 0, []
    start = time.time()

    with open(RESULTS_PATH, "w", encoding="utf-8") as out:
        for i, item in enumerate(items, 1):
            try:
                result = run_pipeline(
                    input_id=item.item_id,
                    original_output=item.output_text,
                    task_type=item.task_type,
                    benchmark_item=item,
                )
                out.write(json.dumps(result.model_dump()) + "\n")
                out.flush()  # crash-safe: completed items are already on disk
                n_ok += 1
            except Exception as e:
                failures.append({"item_id": item.item_id, "error": str(e)})
                print(f"[{i}/{len(items)}] FAILED {item.item_id}: {e}")

            if i % 10 == 0 or i == len(items):
                elapsed = time.time() - start
                remaining = (elapsed / i) * (len(items) - i)
                print(f"[{i}/{len(items)}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

    meta = {
        "mock_critic_a": MOCK_CRITIC_A,
        "mock_critic_b": MOCK_CRITIC_B,
        "mock_critic_c": MOCK_CRITIC_C,
        "ollama_model": OLLAMA_MODEL,
        "sample_source": str(SAMPLE_PATH),
        "total_items": len(items),
        "succeeded": n_ok,
        "failed": len(failures),
        "failures": failures,
        "elapsed_seconds": round(time.time() - start, 1),
        "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {n_ok} succeeded, {len(failures)} failed -> {RESULTS_PATH}")
    print(json.dumps({k: v for k, v in meta.items() if k != "failures"}, indent=2))


if __name__ == "__main__":
    main()