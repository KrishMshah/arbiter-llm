"""
FinalScripts/common.py

Shared loaders, statistics helpers and plot style for the "final" analyses.
Everything reads from results/processed and writes to results/final.

Layout (relative to the repo root, override with ARBITER_ROOT):
  results/processed/   run_sample*.jsonl, run_results_*.jsonl, ...
  results/analysis/    existing arena analysis CSVs
  results/final/plots  every new figure
  results/final/tables every new table (CSV)
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ARBITER_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "results" / "processed"
ANALYSIS = ROOT / "results" / "analysis"
FINAL = ROOT / "results" / "final"
PLOTS = FINAL / "plots"
TABLES = FINAL / "tables"
for _d in (PLOTS, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---- feature layout (must match src/training_data.py FEATURE_NAMES) --------
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
DISAGREEMENT_FEATURES = [
    "score_gap_max", "score_gap_mean", "score_variance", "disagreement_count",
    "disagreement_type_score_divergence", "disagreement_type_issue_miss",
    "disagreement_type_severity_mismatch", "disagreement_type_false_positive",
    "disagreement_type_issue_presence_split",
]
CRITIC_SIGNAL_FEATURES = [
    "score_mean", "critic_confidence_mean", "critic_confidence_min",
    "issue_severity_max", "issue_count_total", "critics_used_count",
]
TASK_FEATURES = [
    "task_type_factual_qa", "task_type_summarisation",
    "task_type_reasoning", "task_type_creative",
]

# Per-critic cost per item, measured from API-reported token usage on the
# 260-item arena run (the 779-item run only has text-reconstructed estimates,
# which undercount because they omit the structured-output scaffolding).
# Filled in by measured_critic_costs().
CRITIC_IDS = ["critic_a", "critic_b", "critic_c"]
CRITIC_LABEL = {
    "critic_a": "GPT-4o-mini",
    "critic_b": "Claude Haiku 4.5",
    "critic_c": "Llama 3.2 3B (local)",
}
DATASET_LABEL = {
    "mt_bench_human": "MT-Bench (human)",
    "summeval": "SummEval",
    "truthfulqa": "TruthfulQA",
    "factscore_labeled": "FActScore (labeled)",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_sample_items():
    items = {}
    for name in ("run_sample.jsonl", "run_sample_expansion.jsonl"):
        for it in load_jsonl(PROCESSED / name):
            items[it["item_id"]] = it
    return items


def load_combined():
    return load_jsonl(PROCESSED / "run_results_combined_costed.jsonl")


def build_dataset():
    """One row per labelled item: 19 stored features, label, dataset, and
    pointers back to the full record (critiques, trace, benchmark item)."""
    items = load_sample_items()
    records = load_combined()
    rows = []
    for rec in records:
        it = items.get(rec["input_id"])
        if it is None or it["human_quality_score"] is None:
            continue
        row = {"item_id": rec["input_id"],
               "dataset": it["dataset_source"],
               "task_type": rec["task_type"],
               "y": float(it["human_quality_score"])}
        row.update({k: float(v) for k, v in rec["ml_features"].items() if k in FEATURE_NAMES})
        rows.append(row)
    df = pd.DataFrame(rows)
    by_id = {r["input_id"]: r for r in records}
    return df, items, by_id


def measured_critic_costs():
    """Mean dollars per item for each critic, from API-reported tokens on
    the arena run. Returns dict critic_id -> dollars per item."""
    pricing = {
        "gpt-4o-mini": (0.00015, 0.00060),
        "claude-haiku-4-5": (0.00100, 0.00500),
        "llama3.2:3b": (0.0, 0.0),
    }
    recs = load_jsonl(PROCESSED / "run_results_arena_reprocessed.jsonl")
    tot = {c: 0.0 for c in CRITIC_IDS}
    for r in recs:
        for c in r["critiques"]:
            pin, pout = pricing[c["model_used"]]
            tot[c["critic_id"]] += (c["input_tokens"] or 0) / 1000 * pin + (c["output_tokens"] or 0) / 1000 * pout
    return {k: v / len(recs) for k, v in tot.items()}


# ---- statistics -------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def bootstrap_ci(values, stat=np.mean, n_boot=2000, seed=SEED, alpha=0.05):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = np.array([stat(values[i]) for i in idx])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def paired_bootstrap_diff(a, b, n_boot=4000, seed=SEED, alpha=0.05):
    """CI for mean(a) - mean(b) over items, resampling items jointly."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float((a - b).mean()), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


# ---- plot style -------------------------------------------------------------
BLUE = "#1f5fa8"
ORANGE = "#d8651f"
GREY = "#6b6b6b"
LIGHTGREY = "#bdbdbd"
GREEN = "#2e7d4f"
RED = "#b23a3a"


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e3e3e3",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
    return plt


def save_fig(fig, filename):
    path = PLOTS / filename
    fig.savefig(path)
    print(f"[plot] {path.relative_to(ROOT)}")
    return path


def save_table(df, filename):
    path = TABLES / filename
    df.to_csv(path, index=False)
    print(f"[table] {path.relative_to(ROOT)}")
    return path