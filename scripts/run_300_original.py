"""scripts/run_300_original.py — original 300-item sample."""
from pathlib import Path
from src.run_pipeline_batch import run_batch

OUT_ROOT = Path("results/processed")
run_batch(
    sample_path=OUT_ROOT / "run_sample.jsonl",
    results_path=OUT_ROOT / "run_results.jsonl",
    meta_path=OUT_ROOT / "run_metadata.json",
    stage_name="300_original",
)