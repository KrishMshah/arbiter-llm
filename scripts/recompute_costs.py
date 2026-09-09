"""scripts/recompute_costs.py — uses the real prompt/response text, not proxies."""
import json
from pathlib import Path
from src.pipeline import PRICING, count_tokens
from src.critics import build_prompt

IN_PATH = Path("results/processed/run_results.jsonl")
OUT_PATH = Path("results/processed/run_results_costed.jsonl")


def cost_for(model_used, input_text, output_text):
    rates = PRICING.get(model_used)
    if rates is None:
        return 0.0
    return (count_tokens(input_text) / 1000) * rates["input_per_1k"] + \
           (count_tokens(output_text) / 1000) * rates["output_per_1k"]


records = []
with open(IN_PATH, encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        breakdown = {}
        for c in rec["critiques"]:
            prompt = build_prompt(rec["original_output"], c["dimensions_evaluated"])
            full_output = json.dumps({
                "dimension_scores": c["dimension_scores"],
                "issues": c["issues"],
                "self_confidence": c["self_confidence"],
                "reasoning": c["reasoning"],
            })
            breakdown[c["critic_id"]] = cost_for(c["model_used"], prompt, full_output)
        breakdown["adjudicator"] = 0.0
        rec["cost_breakdown"] = breakdown
        rec["total_cost_usd"] = sum(breakdown.values())
        records.append(rec)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

total = sum(r["total_cost_usd"] for r in records)
print(f"Recomputed {len(records)} items -> {OUT_PATH}")
print(f"Total run cost: ${total:.4f}")