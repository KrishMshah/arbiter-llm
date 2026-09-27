"""scripts/export_arena_csv.py"""
import csv
import json
from pathlib import Path

OUT_ROOT = Path("results/processed")
RESULTS_PATH = OUT_ROOT / "run_results_arena_reprocessed.jsonl"
PAIR_META_PATH = OUT_ROOT / "run_sample_arena_meta.json"
CSV_PATH = OUT_ROOT / "run_results_arena.csv"

FIELDNAMES = [
    "item_id", "pair_id", "side", "opponent_model", "human_vote_outcome", "task_type",
    "critic_a_mean_score", "critic_a_issue_count", "critic_a_failed",
    "critic_b_mean_score", "critic_b_issue_count", "critic_b_failed",
    "critic_c_mean_score", "critic_c_issue_count", "critic_c_failed",
    "has_disagreement", "disagreement_count", "max_score_gap",
    "ml_predicted_quality_score", "ml_arbitration_confidence", "ml_model_used", "ml_escalate_to_gpt",
    "verdict_quality_score", "verdict_confidence", "verdict_adjudicated", "verdict_adjudicated_by",
    "has_hallucination", "latency_ms",
]


def _build_pair_lookup(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lookup = {}
    for pid, p in meta["pairs"].items():
        lookup[p["item_id_a"]] = {"pair_id": pid, "side": "a", **p["raw_metadata_a"]}
        lookup[p["item_id_b"]] = {"pair_id": pid, "side": "b", **p["raw_metadata_b"]}
    return lookup


def _critic_summary(critiques, cid):
    c = next((c for c in critiques if c["critic_id"] == cid), None)
    if c is None or c.get("critic_failed"):
        return {"mean_score": "", "issue_count": 0 if c else "", "failed": bool(c and c["critic_failed"])}
    scores = list(c["dimension_scores"].values())
    return {"mean_score": round(sum(scores) / len(scores), 3) if scores else "", "issue_count": len(c["issues"]), "failed": False}


def main():
    records = [json.loads(l) for l in RESULTS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    pair_lookup = _build_pair_lookup(PAIR_META_PATH)

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in records:
            pair_info = pair_lookup.get(r["input_id"], {})
            row = {
                "item_id": r["input_id"],
                "pair_id": pair_info.get("pair_id", ""),
                "side": pair_info.get("side", ""),
                "opponent_model": pair_info.get("opponent_model", ""),
                "human_vote_outcome": pair_info.get("outcome", ""),
                "task_type": r["task_type"],
            }
            for cid in ["critic_a", "critic_b", "critic_c"]:
                s = _critic_summary(r["critiques"], cid)
                row[f"{cid}_mean_score"] = s["mean_score"]
                row[f"{cid}_issue_count"] = s["issue_count"]
                row[f"{cid}_failed"] = s["failed"]
            row["has_disagreement"] = r["disagreement_matrix"]["has_disagreement"]
            row["disagreement_count"] = r["disagreement_matrix"]["disagreement_count"]
            row["max_score_gap"] = r["disagreement_matrix"]["max_score_gap"]
            row["ml_predicted_quality_score"] = r["ml_arbitrator_output"]["predicted_quality_score"]
            row["ml_arbitration_confidence"] = r["ml_arbitrator_output"]["arbitration_confidence"]
            row["ml_model_used"] = r["ml_arbitrator_output"]["model_used"]
            row["ml_escalate_to_gpt"] = r["ml_arbitrator_output"]["escalate_to_gpt"]
            row["verdict_quality_score"] = r["verdict"]["quality_score"]
            row["verdict_confidence"] = r["verdict"]["confidence"]
            row["verdict_adjudicated"] = r["verdict"]["adjudicated"]
            row["verdict_adjudicated_by"] = r["verdict"]["adjudicated_by"]
            row["has_hallucination"] = r["hallucination_trace"]["has_hallucination"]
            row["latency_ms"] = r["latency_ms"]
            writer.writerow(row)

    print(f"{len(records)} rows -> {CSV_PATH}")


if __name__ == "__main__":
    main()