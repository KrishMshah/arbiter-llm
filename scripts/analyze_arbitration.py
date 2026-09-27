"""
scripts/analyze_arbitration.py

Diagnoses the ML arbitration cascade on an already-produced results file.
Reads results/processed/run_results_arena_reprocessed.jsonl -- no critic
recall, no adjudicator recall, zero cost.

Writes:
  results/analysis/arbitration_per_item.csv
  results/analysis/arbitration_summary.csv
  results/analysis/confidence_distribution.png
  results/analysis/resolution_path.png
  results/analysis/cost_by_path.png

Usage:
  python -m scripts.analyze_arbitration
"""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ML_CONFIDENCE_THRESHOLD

INPUT_PATH = Path("results/processed/run_results_arena_reprocessed.jsonl")
OUT_DIR = Path("results/analysis")


def load_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records(INPUT_PATH)
    n = len(records)
    if n == 0:
        print(f"[analyze] no records found in {INPUT_PATH}")
        return

    model_used_counts = Counter()
    confidences = []
    escalated = 0
    rows = []

    for r in records:
        ml = r["ml_arbitrator_output"]
        model_used_counts[ml["model_used"]] += 1
        conf = ml["arbitration_confidence"]
        confidences.append(conf)
        esc = bool(ml["escalate_to_gpt"])
        escalated += int(esc)
        rows.append(
            {
                "input_id": r["input_id"],
                "task_type": r["task_type"],
                "model_used": ml["model_used"],
                "arbitration_confidence": conf,
                "escalated": esc,
                "predicted_quality_score": ml["predicted_quality_score"],
                "final_quality_score": r["verdict"]["quality_score"],
                "adjudicated_by": r["verdict"]["adjudicated_by"],
                "total_cost_usd": r["total_cost_usd"],
            }
        )

    resolved_by_ml = n - escalated
    sorted_conf = sorted(confidences)
    median_conf = sorted_conf[n // 2]

    # ---- per-item CSV ----
    per_item_path = OUT_DIR / "arbitration_per_item.csv"
    with open(per_item_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- summary CSV ----
    ml_cost = sum(r["total_cost_usd"] for r in rows if not r["escalated"])
    adj_cost = sum(r["total_cost_usd"] for r in rows if r["escalated"])
    summary_path = OUT_DIR / "arbitration_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_items", n])
        writer.writerow(["resolved_by_ml_arbitrator", resolved_by_ml])
        writer.writerow(["escalated_to_adjudicator", escalated])
        writer.writerow(["resolved_pct", round(100 * resolved_by_ml / n, 2)])
        writer.writerow(["escalated_pct", round(100 * escalated / n, 2)])
        writer.writerow(["ml_confidence_threshold", ML_CONFIDENCE_THRESHOLD])
        writer.writerow(["mean_confidence", round(sum(confidences) / n, 4)])
        writer.writerow(["median_confidence", round(median_conf, 4)])
        writer.writerow(["min_confidence", round(min(confidences), 4)])
        writer.writerow(["max_confidence", round(max(confidences), 4)])
        for model, count in model_used_counts.items():
            writer.writerow([f"model_used:{model}", count])
        writer.writerow(["cost_ml_resolved_usd", round(ml_cost, 4)])
        writer.writerow(["cost_escalated_usd", round(adj_cost, 4)])
        writer.writerow(["cost_total_usd", round(ml_cost + adj_cost, 4)])

    print(
        f"[analyze] {n} items | resolved by ML: {resolved_by_ml} "
        f"({100*resolved_by_ml/n:.1f}%) | escalated: {escalated} "
        f"({100*escalated/n:.1f}%)"
    )
    print(f"[analyze] ML_CONFIDENCE_THRESHOLD = {ML_CONFIDENCE_THRESHOLD}")
    print(
        f"[analyze] confidence: mean={sum(confidences)/n:.4f} "
        f"median={median_conf:.4f} min={min(confidences):.4f} "
        f"max={max(confidences):.4f}"
    )
    print(f"[analyze] model_used distribution: {dict(model_used_counts)}")
    print(f"[analyze] wrote {per_item_path}")
    print(f"[analyze] wrote {summary_path}")

    # ---- plot 1: confidence histogram vs threshold ----
    plt.figure(figsize=(7, 4.5))
    plt.hist(confidences, bins=20, color="#4C72B0", edgecolor="white")
    plt.axvline(
        ML_CONFIDENCE_THRESHOLD,
        color="#C44E52",
        linestyle="--",
        label=f"threshold = {ML_CONFIDENCE_THRESHOLD}",
    )
    plt.xlabel("arbitration_confidence")
    plt.ylabel("items")
    plt.title("Arbitration confidence distribution vs. escalation threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confidence_distribution.png", dpi=150)
    plt.close()

    # ---- plot 2: resolution path ----
    plt.figure(figsize=(5, 4.5))
    bars = plt.bar(
        ["Resolved by ML", "Escalated to adjudicator"],
        [resolved_by_ml, escalated],
        color=["#55A868", "#C44E52"],
    )
    for bar, v in zip(bars, [resolved_by_ml, escalated]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            v + n * 0.01,
            str(v),
            ha="center",
        )
    plt.ylabel("items")
    plt.title(f"Resolution path (n={n})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "resolution_path.png", dpi=150)
    plt.close()

    # ---- plot 3: cost by resolution path ----
    plt.figure(figsize=(5, 4.5))
    plt.bar(
        ["ML-resolved", "Adjudicator-escalated"],
        [ml_cost, adj_cost],
        color=["#55A868", "#C44E52"],
    )
    plt.ylabel("USD")
    plt.title("Cost by resolution path")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cost_by_path.png", dpi=150)
    plt.close()

    print(f"[analyze] wrote plots to {OUT_DIR}")


if __name__ == "__main__":
    main()