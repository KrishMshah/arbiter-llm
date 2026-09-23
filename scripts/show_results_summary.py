"""
scripts/show_results_summary.py

Instant headline stats from the already-completed real run. No API
calls, no re-running the pipeline — just reads what's already on disk.
"""

import json
from pathlib import Path

RESULTS_PATH = Path("results/processed/run_results_costed.jsonl")
SAMPLE_PATH = Path("results/processed/run_sample.jsonl")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def flagged(crit):
    return any(i["dimension"] == "factual_accuracy" for i in crit["issues"])


results = load_jsonl(RESULTS_PATH)
sample = {s["item_id"]: s for s in load_jsonl(SAMPLE_PATH)}

n_disagree = sum(1 for r in results if r["disagreement_matrix"]["has_disagreement"])
total_cost = sum(r["total_cost_usd"] for r in results)

print(f"Items run: {len(results)}")
print(f"Disagreement rate: {n_disagree}/{len(results)} ({100*n_disagree/len(results):.1f}%)")
print(f"Total real cost: ${total_cost:.4f}")

tqa = [r for r in results if sample[r["input_id"]]["dataset_source"] == "truthfulqa"]
for cid, label in [("critic_a", "Critic A"), ("critic_b", "Critic B"), ("critic_c", "Critic C")]:
    correct = 0
    for r in tqa:
        truth = sample[r["input_id"]]["ground_truth_label"] == "untruthful"
        crit = next(c for c in r["critiques"] if c["critic_id"] == cid)
        if crit["critic_failed"]:
            continue
        if flagged(crit) == truth:
            correct += 1
    print(f"{label} TruthfulQA accuracy: {correct}/{len(tqa)} ({100*correct/len(tqa):.1f}%)")