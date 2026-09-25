"""
src/run_real_sample.py

Runs run_sample_expansion.jsonl  Writes incrementally so a crash
mid-run doesn't lose completed items, and is resumable: re-running this
script skips whatever's already in RESULTS_PATH and only processes what's
left, appending rather than overwriting.

Safety net: if the same critic fails CONSECUTIVE_FAILURE_LIMIT times in a
row (out-of-funds, revoked key, dead network -- pipeline.py's per-critic
catch means we never see the raw exception, only a failed critique), the
run halts immediately rather than silently burning through the rest of
the sample recording failures. Same breaker for repeated whole-pipeline
exceptions. Top up funds / fix whatever broke, re-run this exact script,
it picks up where it stopped.
"""

import json
import time
from pathlib import Path

from src.config import MOCK_CRITIC_A, MOCK_CRITIC_B, MOCK_CRITIC_C, OLLAMA_MODEL
from src.pipeline import run_pipeline
from src.schemas import BenchmarkItem

OUT_ROOT = Path("results/processed")
SAMPLE_PATH = OUT_ROOT / "run_sample_expansion.jsonl"
RESULTS_PATH = OUT_ROOT / "run_results_expansion.jsonl"  # separate file -- never touches the original 300's run_results.jsonl
META_PATH = OUT_ROOT / "run_metadata_expansion.json"

CONSECUTIVE_FAILURE_LIMIT = 3
PROGRESS_INTERVAL = 10

# Best-effort guess at what an out-of-funds error looks like -- haven't seen
# the real message yet. First time this actually triggers, paste it and
# this list gets tightened. The consecutive-failure count above is the real
# safety net; this only makes the halt message more specific when it can.
_BUDGET_EXHAUSTION_SIGNATURES = [
    "insufficient_quota", "insufficient quota", "credit balance",
    "billing_hard_limit", "billing hard limit", "exceeded your current quota",
    "quota_exceeded", "quota exceeded", "invalid_api_key",
]


def _looks_like_budget_exhaustion(reason: str) -> bool:
    reason_lower = (reason or "").lower()
    return any(sig in reason_lower for sig in _BUDGET_EXHAUSTION_SIGNATURES)


def load_sample() -> list[BenchmarkItem]:
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        return [BenchmarkItem(**json.loads(line)) for line in f]


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["input_id"] for line in f if line.strip()}


def main():
    all_items = load_sample()
    completed_ids = load_completed_ids(RESULTS_PATH)
    items = [item for item in all_items if item.item_id not in completed_ids]

    if completed_ids:
        print(f"Resuming: {len(completed_ids)}/{len(all_items)} already done, {len(items)} remaining.")
    else:
        print(f"Fresh run: {len(all_items)} items through the full pipeline (all 3 critics real)...")

    if not items:
        print("Nothing left to do -- all items already completed.")
        return

    n_ok, failures = 0, []
    consecutive_critic_failures = {"critic_a": 0, "critic_b": 0, "critic_c": 0}
    consecutive_pipeline_exceptions = 0
    halted_early = False
    halt_reason = None
    start = time.time()

    mode = "a" if completed_ids else "w"
    with open(RESULTS_PATH, mode, encoding="utf-8") as out:
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
                print(f"[{i}/{len(items)}] FAILED {item.item_id}: {e}")
                consecutive_pipeline_exceptions += 1
                if consecutive_pipeline_exceptions >= CONSECUTIVE_FAILURE_LIMIT:
                    halt_reason = f"whole pipeline raised {consecutive_pipeline_exceptions} times in a row. Last error: {e}"
                    halted_early = True

            if i % PROGRESS_INTERVAL == 0 or i == len(items) or halted_early:
                elapsed = time.time() - start
                remaining = (elapsed / i) * (len(items) - i)
                print(f"[{i}/{len(items)}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining "
                    f"({len(completed_ids) + n_ok}/{len(all_items)} total done)")

            if halted_early:
                print(f"\nSTOPPED EARLY: {halt_reason}")
                print(f"{len(completed_ids) + n_ok}/{len(all_items)} total items completed and saved.")
                print(f"Fix the API key/funds, then re-run this exact script -- it resumes automatically "
                    f"from {RESULTS_PATH}, nothing already done is repeated.")
                break

    meta = {
        "mock_critic_a": MOCK_CRITIC_A,
        "mock_critic_b": MOCK_CRITIC_B,
        "mock_critic_c": MOCK_CRITIC_C,
        "ollama_model": OLLAMA_MODEL,
        "sample_source": str(SAMPLE_PATH),
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
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if not halted_early:
        print(f"\nDone: {len(completed_ids) + n_ok}/{len(all_items)} total items completed -> {RESULTS_PATH}")
    print(json.dumps({k: v for k, v in meta.items() if k != "failures_this_session"}, indent=2))


if __name__ == "__main__":
    main()