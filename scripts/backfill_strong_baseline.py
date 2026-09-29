"""
scripts/backfill_strong_baseline.py
Reads:  results/processed/run_results_arena_reprocessed.jsonl
Writes: results/processed/strong_baseline_backfill.jsonl

Usage: python -m scripts.backfill_strong_baseline
"""

import json
from pathlib import Path

from src.schemas import ArbitrationResult
from src.adjudicator import run_adjudicator, retrieve_evidence
from src.pipeline import cost_from_tokens

INPUT_PATH = Path("results/processed/run_results_arena_reprocessed.jsonl")
OUT_PATH = Path("results/processed/strong_baseline_backfill.jsonl")


def load_results(path):
    with open(path, encoding="utf-8") as f:
        return [ArbitrationResult(**json.loads(line)) for line in f if line.strip()]


def load_done_ids(path):
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["input_id"])
    return done


def main():
    results = load_results(INPUT_PATH)
    to_backfill = [r for r in results if not r.verdict.adjudicated]
    done_ids = load_done_ids(OUT_PATH)
    remaining = [r for r in to_backfill if r.input_id not in done_ids]

    print(f"[backfill] {len(to_backfill)} need backfill, {len(done_ids)} already done, {len(remaining)} to run")

    total_cost = 0.0
    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for i, r in enumerate(remaining, 1):
            try:
                retrieve_evidence(r.original_output)  # mirrors pipeline.py's call pattern
                verdict, input_tokens, output_tokens = run_adjudicator(
                    r.original_output, r.critiques, r.disagreement_matrix, r.ml_arbitrator_output,
                )
                cost = cost_from_tokens("gpt-5.6-terra", input_tokens, output_tokens, is_mock=False)
                total_cost += cost
                out.write(json.dumps({
                    "input_id": r.input_id,
                    "strong_quality_score": verdict.quality_score,
                    "strong_confidence": verdict.confidence,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": round(cost, 6),
                }) + "\n")
                out.flush()
                print(f"  [{i}/{len(remaining)}] {r.input_id}: strong_quality_score={verdict.quality_score} (${cost:.4f})")
            except Exception as e:
                print(f"  [{i}/{len(remaining)}] {r.input_id}: FAILED -- {e} (re-run script to retry)")

    print(f"\n[backfill] total new cost this run: ${total_cost:.4f}")


if __name__ == "__main__":
    main()