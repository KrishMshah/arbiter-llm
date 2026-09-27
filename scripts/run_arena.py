"""scripts/run_arena.py — 130-pair (260-item) Chatbot Arena sample, for
the Zheng et al. agreement-rate comparison."""
from pathlib import Path
from src.run_pipeline_batch import run_batch

OUT_ROOT = Path("results/processed")
run_batch(
    sample_path=OUT_ROOT / "run_sample_arena.jsonl",
    results_path=OUT_ROOT / "run_results_arena.jsonl",
    meta_path=OUT_ROOT / "run_metadata_arena.json",
    stage_name="arena",
)