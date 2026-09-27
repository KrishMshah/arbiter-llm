"""scripts/reprocess_arena.py"""
from pathlib import Path
from src.reprocess_arbitration import reprocess

OUT_ROOT = Path("results/processed")
reprocess(
    input_path=OUT_ROOT / "run_results_arena.jsonl",
    output_path=OUT_ROOT / "run_results_arena_reprocessed.jsonl",
    stage_name="arena_reprocess",
)