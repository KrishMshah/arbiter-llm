"""scripts/rerun_failed_items.py"""
import json
from pathlib import Path
from src.pipeline import run_pipeline
from src.schemas import BenchmarkItem

RESULTS_PATH = Path("results/processed/run_results.jsonl")
SAMPLE_PATH = Path("results/processed/run_sample.jsonl")
FAILED_IDS = {"mtbench_human_001989_a", "mtbench_human_002671_a"}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


sample_by_id = {i["item_id"]: i for i in load_jsonl(SAMPLE_PATH)}
results = load_jsonl(RESULTS_PATH)

for i, rec in enumerate(results):
    if rec["input_id"] in FAILED_IDS:
        item = BenchmarkItem(**sample_by_id[rec["input_id"]])
        print(f"Re-running {item.item_id}...")
        results[i] = run_pipeline(
            input_id=item.item_id, original_output=item.output_text,
            task_type=item.task_type, benchmark_item=item,
        ).model_dump()

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print("Patched 2 items back into", RESULTS_PATH)