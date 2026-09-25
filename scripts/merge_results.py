"""
scripts/merge_results.py

Combines the original 300-item run with the 479-item Phase 2b expansion
into a single validated file for downstream training/analysis, plus a
flattened CSV summary for human review (professor, etc).

schemas.py wasn't available when either run happened, so this is the
first point either file gets checked against the real schema rather than
just "did json.loads() not throw."
"""

import csv
import json
from pathlib import Path

from src.schemas import ArbitrationResult

OUT_ROOT = Path("results/processed")
SOURCES = [
    OUT_ROOT / "run_results.jsonl",            # original 300
    OUT_ROOT / "run_results_expansion.jsonl",   # new 479
]
COMBINED_JSONL_PATH = OUT_ROOT / "run_results_combined.jsonl"
COMBINED_CSV_PATH = OUT_ROOT / "run_results_combined.csv"

CSV_FIELDNAMES = [
    "item_id", "dataset_source", "task_type",
    "critic_a_mean_score", "critic_a_confidence", "critic_a_issue_count", "critic_a_failed",
    "critic_b_mean_score", "critic_b_confidence", "critic_b_issue_count", "critic_b_failed",
    "critic_c_mean_score", "critic_c_confidence", "critic_c_issue_count", "critic_c_failed",
    "has_disagreement", "disagreement_count", "max_score_gap",
    "ml_predicted_quality_score", "ml_arbitration_confidence", "ml_model_used", "ml_escalate_to_gpt",
    "verdict_quality_score", "verdict_confidence", "verdict_adjudicated", "verdict_adjudicated_by",
    "has_hallucination", "hallucination_propagation_depth", "hallucination_affected_sentences",
    "latency_ms",
]

_ID_PREFIX_TO_SOURCE = [
    ("factscore_labeled", "factscore_labeled"),
    ("factscore_unlabeled", "factscore_unlabeled"),
    ("truthfulqa", "truthfulqa"),
    ("summeval", "summeval"),
    ("mtbench_human", "mt_bench_human"),
    ("mt_bench_human", "mt_bench_human"),
    ("mtbench_gpt4", "mt_bench_gpt4_pair"),
    ("chatbot_arena", "chatbot_arena"),
    ("arena", "chatbot_arena"),
]


def _dataset_source_from_id(item_id: str) -> str:
    for prefix, source in _ID_PREFIX_TO_SOURCE:
        if item_id.startswith(prefix):
            return source
    return "unknown"


def _critic_summary(critiques, critic_id: str) -> dict:
    c = next((c for c in critiques if c.critic_id == critic_id), None)
    if c is None:
        return {"mean_score": "", "confidence": "", "issue_count": "", "failed": ""}
    if c.critic_failed:
        return {"mean_score": "", "confidence": "", "issue_count": 0, "failed": True}
    scores = list(c.dimension_scores.values())
    mean_score = round(sum(scores) / len(scores), 3) if scores else ""
    return {
        "mean_score": mean_score,
        "confidence": c.self_confidence if c.self_confidence is not None else "",
        "issue_count": len(c.issues),
        "failed": False,
    }


def flatten_for_csv(r: ArbitrationResult) -> dict:
    row = {
        "item_id": r.input_id,
        "dataset_source": _dataset_source_from_id(r.input_id),
        "task_type": r.task_type,
    }
    for cid in ["critic_a", "critic_b", "critic_c"]:
        s = _critic_summary(r.critiques, cid)
        row[f"{cid}_mean_score"] = s["mean_score"]
        row[f"{cid}_confidence"] = s["confidence"]
        row[f"{cid}_issue_count"] = s["issue_count"]
        row[f"{cid}_failed"] = s["failed"]

    row["has_disagreement"] = r.disagreement_matrix.has_disagreement
    row["disagreement_count"] = r.disagreement_matrix.disagreement_count
    row["max_score_gap"] = r.disagreement_matrix.max_score_gap

    row["ml_predicted_quality_score"] = r.ml_arbitrator_output.predicted_quality_score
    row["ml_arbitration_confidence"] = r.ml_arbitrator_output.arbitration_confidence
    row["ml_model_used"] = r.ml_arbitrator_output.model_used
    row["ml_escalate_to_gpt"] = r.ml_arbitrator_output.escalate_to_gpt

    row["verdict_quality_score"] = r.verdict.quality_score
    row["verdict_confidence"] = r.verdict.confidence
    row["verdict_adjudicated"] = r.verdict.adjudicated
    row["verdict_adjudicated_by"] = r.verdict.adjudicated_by

    row["has_hallucination"] = r.hallucination_trace.has_hallucination
    row["hallucination_propagation_depth"] = r.hallucination_trace.propagation_depth
    row["hallucination_affected_sentences"] = r.hallucination_trace.total_affected_sentences

    row["latency_ms"] = r.latency_ms
    return row


def load_and_validate(path: Path) -> list[ArbitrationResult]:
    validated = []
    errors = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                validated.append(ArbitrationResult(**json.loads(line)))
            except Exception as e:
                errors.append((i, str(e)))
    if errors:
        print(f"  {len(errors)} validation failures in {path.name}:")
        for line_no, err in errors[:5]:
            print(f"    line {line_no}: {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")
    return validated


def main():
    all_results: list[ArbitrationResult] = []
    seen_ids: set[str] = set()
    duplicates: list[str] = []

    for path in SOURCES:
        with open(path, encoding="utf-8") as f:
            line_count = sum(1 for line in f if line.strip())
        print(f"Loading {path}...")
        results = load_and_validate(path)
        print(f"  {len(results)}/{line_count} lines valid")

        for r in results:
            if r.input_id in seen_ids:
                duplicates.append(r.input_id)
                continue
            seen_ids.add(r.input_id)
            all_results.append(r)

    if duplicates:
        print(f"\nWARNING: {len(duplicates)} duplicate item_ids skipped (kept first occurrence):")
        print(f"  {duplicates[:10]}")

    with open(COMBINED_JSONL_PATH, "w", encoding="utf-8") as out:
        for r in all_results:
            out.write(r.model_dump_json() + "\n")
    print(f"\nCombined JSONL: {len(all_results)} unique, validated items -> {COMBINED_JSONL_PATH}")

    with open(COMBINED_CSV_PATH, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for r in all_results:
            writer.writerow(flatten_for_csv(r))
    print(f"Combined CSV: {len(all_results)} rows -> {COMBINED_CSV_PATH}")

    print("\nNOTE: cost fields deliberately excluded from the CSV -- old "
        "estimate_cost() bug, not fixed yet. ml_model_used will read "
        "'stub' for every row until arbitrator.py is actually trained.")


if __name__ == "__main__":
    main()