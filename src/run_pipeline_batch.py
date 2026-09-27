"""
src/run_pipeline_batch.py

Shared engine for running a benchmark sample through the full pipeline.
All 3 critics real. Writes incrementally (crash-safe) and resumes
automatically on re-run. Used by scripts/run_*.py -- each stage (original
300, 480-item expansion, 130-pair Arena sample) calls this with its own
paths, so the run logic itself lives in exactly one place.
"""

import json
import time
from pathlib import Path

from src.config import MOCK_CRITIC_A, MOCK_CRITIC_B, MOCK_CRITIC_C, OLLAMA_MODEL
from src.pipeline import run_pipeline
from src.schemas import BenchmarkItem

CONSECUTIVE_FAILURE_LIMIT = 3
PROGRESS_INTERVAL = 10

_BUDGET_EXHAUSTION_SIGNATURES = [
    "insufficient_quota", "insufficient quota", "credit balance",
    "billing_hard_limit", "billing hard limit", "exceeded your current quota",
    "quota_exceeded", "quota exceeded", "invalid_api_key",
]


def _looks_like_budget_exhaustion(reason: str) -> bool:
    reason_lower = (reason or "").lower()
    return any(sig in reason_lower for sig in _BUDGET_EXHAUSTION_SIGNATURES)


def _load_sample(sample_path: Path) -> list[BenchmarkItem]:
    with open(sample_path, encoding="utf-8") as f:
        return [BenchmarkItem(**json.loads(line)) for line in f]


def _load_completed_ids(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    with open(results_path, encoding="utf-8") as f:
        return {json.loads(line)["input_id"] for line in f if line.strip()}


def run_batch(sample_path: Path, results_path: Path, meta_path: Path, stage_name: str):
    all_items = _load_sample(sample_path)
    completed_ids = _load_completed_ids(results_path)
    items = [item for item in all_items if item.item_id not in completed_ids]

    if completed_ids:
        print(f"[{stage_name}] Resuming: {len(completed_ids)}/{len(all_items)} already done, {len(items)} remaining.")
    else:
        print(f"[{stage_name}] Fresh run: {len(all_items)} items through the full pipeline...")

    if not items:
        print(f"[{stage_name}] Nothing left to do -- all items already completed.")
        return

    n_ok, failures = 0, []
    consecutive_critic_failures = {"critic_a": 0, "critic_b": 0, "critic_c": 0}
    consecutive_pipeline_exceptions = 0
    halted_early = False
    halt_reason = None
    start = time.time()

    mode = "a" if completed_ids else "w"
    with open(results_path, mode, encoding="utf-8") as out:
        for i, item in enumerate(items, 1):
            try:
                result = run_pipeline(
                    input_id=item.item_id,
                    original_output=item.output_text,
                    task_type=item.task_type,
                    benchmark_item=item,
                )
                out.write(json.dumps(result.model_dump()) + "\n")
                out.flush()
                n_ok += 1
                consecutive_pipeline_exceptions = 0

                for critique in result.critiques:
                    cid = critique.critic_id
                    if critique.critic_failed:
                        consecutive_critic_failures[cid] += 1
                    else:
                        consecutive_critic_failures[cid] = 0

                for cid, count in consecutive_critic_failures.items():
                    if count >= CONSECUTIVE_FAILURE_LIMIT:
                        last_failed = next(
                            (c for c in result.critiques if c.critic_id == cid and c.critic_failed),
                            None,
                        )
                        reason = last_failed.failure_reason if last_failed else "unknown"
                        budget_guess = " -- looks like an API budget/quota issue" if _looks_like_budget_exhaustion(reason) else ""
                        halt_reason = f"{cid} failed {count} times in a row{budget_guess}. Last error: {reason}"
                        halted_early = True
                        break

            except Exception as e:
                failures.append({"item_id": item.item_id, "error": str(e)})
                print(f"[{stage_name}] [{i}/{len(items)}] FAILED {item.item_id}: {e}")
                consecutive_pipeline_exceptions += 1
                if consecutive_pipeline_exceptions >= CONSECUTIVE_FAILURE_LIMIT:
                    halt_reason = f"whole pipeline raised {consecutive_pipeline_exceptions} times in a row. Last error: {e}"
                    halted_early = True

            if i % PROGRESS_INTERVAL == 0 or i == len(items) or halted_early:
                elapsed = time.time() - start
                remaining = (elapsed / i) * (len(items) - i)
                print(f"[{stage_name}] [{i}/{len(items)}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining "
                      f"({len(completed_ids) + n_ok}/{len(all_items)} total done)")

            if halted_early:
                print(f"\n[{stage_name}] STOPPED EARLY: {halt_reason}")
                print(f"{len(completed_ids) + n_ok}/{len(all_items)} total items completed and saved.")
                print(f"Fix the issue, then re-run this exact script -- it resumes automatically "
                      f"from {results_path}, nothing already done is repeated.")
                break

    meta = {
        "stage": stage_name,
        "mock_critic_a": MOCK_CRITIC_A,
        "mock_critic_b": MOCK_CRITIC_B,
        "mock_critic_c": MOCK_CRITIC_C,
        "ollama_model": OLLAMA_MODEL,
        "sample_source": str(sample_path),
        "total_items_in_sample": len(all_items),
        "completed_before_this_session": len(completed_ids),
        "succeeded_this_session": n_ok,
        "failed_this_session": len(failures),
        "total_completed": len(completed_ids) + n_ok,
        "halted_early": halted_early,
        "halt_reason": halt_reason,
        "failures_this_session": failures,
        "elapsed_seconds_this_session": round(time.time() - start, 1),
        "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if not halted_early:
        print(f"\n[{stage_name}] Done: {len(completed_ids) + n_ok}/{len(all_items)} total -> {results_path}")
    print(json.dumps({k: v for k, v in meta.items() if k != "failures_this_session"}, indent=2))