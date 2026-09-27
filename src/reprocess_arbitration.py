"""
src/reprocess_arbitration.py

Recomputes arbitrator + adjudicator for an already-collected run, reusing
stored critic outputs -- no critic recall, zero Anthropic/Haiku cost.
Resumable; halts after repeated adjudicator failures rather than burning
through the file silently.
"""

import json
import time
from collections import Counter
from pathlib import Path

from src.arbitrator import run_arbitrator
from src.adjudicator import run_adjudicator, retrieve_evidence
from src.pipeline import cost_from_tokens
from src.schemas import CritiqueOutput, DisagreementMatrix, MLFeatures, Verdict, ArbitrationResult

CONSECUTIVE_FAILURE_LIMIT = 3
PROGRESS_INTERVAL = 10


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["input_id"] for line in f if line.strip()}


def _ml_only_verdict(ml_output, critiques: list[CritiqueOutput]) -> Verdict:
    all_issues = [i for c in critiques for i in c.issues]
    confidence_1_5 = max(1, min(5, round(1 + ml_output.arbitration_confidence * 4)))
    quality_1_10 = max(1, min(10, round(ml_output.predicted_quality_score)))
    return Verdict(
        quality_score=quality_1_10, confidence=confidence_1_5,
        confirmed_issues=all_issues, dismissed_flags=[],
        adjudicated=False, adjudicated_by="ml_arbitrator", adjudicator_reasoning=None,
    )


def reprocess(input_path: Path, output_path: Path, stage_name: str):
    all_records = _load_jsonl(input_path)
    completed_ids = _load_completed_ids(output_path)
    records = [r for r in all_records if r["input_id"] not in completed_ids]

    print(f"[{stage_name}] {len(completed_ids)}/{len(all_records)} already done, {len(records)} to process.")
    if not records:
        return

    consecutive_failures = 0
    n_ok, n_escalated = 0, 0
    model_used_counts = Counter()
    halted, halt_reason = False, None
    start = time.time()

    mode = "a" if completed_ids else "w"
    with open(output_path, mode, encoding="utf-8") as out:
        for i, r in enumerate(records, 1):
            critiques = [CritiqueOutput(**c) for c in r["critiques"]]
            disagreement_matrix = DisagreementMatrix(**r["disagreement_matrix"])
            ml_features = MLFeatures(**r["ml_features"])

            ml_output = run_arbitrator(ml_features)
            model_used_counts[ml_output.model_used] += 1
            cost_breakdown = dict(r["cost_breakdown"])  # critic costs untouched

            try:
                if ml_output.escalate_to_gpt:
                    n_escalated += 1
                    verdict, in_tok, out_tok = run_adjudicator(
                        r["original_output"], critiques, disagreement_matrix, ml_output,
                    )
                    cost_breakdown["adjudicator"] = cost_from_tokens("gpt-5.6-terra", in_tok, out_tok, is_mock=False)
                    retrieved_evidence = retrieve_evidence(r["original_output"])
                else:
                    verdict = _ml_only_verdict(ml_output, critiques)
                    cost_breakdown["adjudicator"] = 0.0
                    retrieved_evidence = None
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                print(f"[{stage_name}] [{i}/{len(records)}] FAILED {r['input_id']}: {e}")
                if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                    halt_reason = f"{consecutive_failures} adjudicator failures in a row: {e}"
                    halted = True
                    break
                continue

            new_result = ArbitrationResult(
                input_id=r["input_id"], task_type=r["task_type"], original_output=r["original_output"],
                routing_decision=r["routing_decision"], critiques=critiques,
                disagreement_matrix=disagreement_matrix, ml_features=ml_features,
                ml_arbitrator_output=ml_output, verdict=verdict,
                hallucination_trace=r["hallucination_trace"], cost_breakdown=cost_breakdown,
                total_cost_usd=round(sum(cost_breakdown.values()), 6), latency_ms=r["latency_ms"],
            )
            out.write(new_result.model_dump_json() + "\n")
            out.flush()
            n_ok += 1

            if i % PROGRESS_INTERVAL == 0 or i == len(records):
                print(f"[{stage_name}] [{i}/{len(records)}] {n_escalated} escalated so far")

    if halted:
        print(f"\n[{stage_name}] STOPPED: {halt_reason}. Re-run to resume.")
    else:
        print(f"\n[{stage_name}] Done: {n_ok} processed -> {output_path}")
        print(f"model_used distribution this session: {dict(model_used_counts)}")
        print(f"Escalated to real adjudicator: {n_escalated}/{len(records)}")