"""
src/training_data.py

Shared data loader. No xgboost/lightgbm imports here -- safe to import
from either training script independently.
"""

import json
from pathlib import Path

import numpy as np

from src.arbitrator import features_to_vector
from src.schemas import MLFeatures

RESULTS_PATH = Path("results/processed/run_results_combined_costed.jsonl")
BENCHMARK_ITEM_PATHS = [
    Path("results/processed/run_sample.jsonl"),
    Path("results/processed/run_sample_expansion.jsonl"),
]
SEED = 42
TEST_SIZE = 0.2

FEATURE_NAMES = [
    "score_gap_max", "score_gap_mean", "score_variance", "score_mean",
    "disagreement_count",
    "disagreement_type_score_divergence", "disagreement_type_issue_miss",
    "disagreement_type_severity_mismatch", "disagreement_type_false_positive",
    "disagreement_type_issue_presence_split",
    "critic_confidence_mean", "critic_confidence_min",
    "issue_severity_max", "issue_count_total", "critics_used_count",
    "task_type_factual_qa", "task_type_summarisation",
    "task_type_reasoning", "task_type_creative",
]


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _load_benchmark_items(paths: list[Path]) -> dict[str, dict]:
    items = {}
    for path in paths:
        for item in _load_jsonl(path):
            items[item["item_id"]] = item
    return items


def build_training_data():
    """X = 19-feature vector, y = human_quality_score, groups = dataset_source."""
    results = _load_jsonl(RESULTS_PATH)
    benchmark_items = _load_benchmark_items(BENCHMARK_ITEM_PATHS)

    X, y, groups = [], [], []
    skipped_no_score = 0

    for r in results:
        bi = benchmark_items.get(r["input_id"])
        if bi is None:
            continue
        score = bi.get("human_quality_score")
        if score is None:
            skipped_no_score += 1
            continue

        features = MLFeatures(**r["ml_features"])
        X.append(features_to_vector(features))
        y.append(score)
        groups.append(bi.get("dataset_source", "unknown"))

    print(f"Training data: {len(X)} items usable, {skipped_no_score} skipped (no human_quality_score)")
    return np.array(X), np.array(y), groups