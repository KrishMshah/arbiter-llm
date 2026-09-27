"""scripts/run_480_expansion.py — 480-item scale-up (300 -> 780)."""
from pathlib import Path
from src.run_pipeline_batch import run_batch

OUT_ROOT = Path("results/processed")
run_batch(
    sample_path=OUT_ROOT / "run_sample_expansion.jsonl",
    results_path=OUT_ROOT / "run_results_expansion.jsonl",
    meta_path=OUT_ROOT / "run_metadata_expansion.json",
    stage_name="480_expansion",
)