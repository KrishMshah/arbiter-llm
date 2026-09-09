"""
src/manifest.py

Pins exact dataset file hashes + processed item counts. Run once after
benchmark.py's build_all() so every reported number is reproducible.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path("benchmark_data")
OUT_ROOT = Path("results/processed")

RAW_FILES = {
    "chatbot_arena": [DATA_ROOT / "chatbot_arena" / "chatbot_arena_train.parquet"],
    "mt_bench_human": [DATA_ROOT / "mt_bench" / "mt_bench_human.parquet"],
    "mt_bench_gpt4_pair": [DATA_ROOT / "mt_bench" / "mt_bench_gpt4_pair.parquet"],
    "truthfulqa": [DATA_ROOT / "truthfulqa" / "truthfulqa_validation.parquet"],
    "summeval": [DATA_ROOT / "summeval" / "summeval_model_annotations.jsonl"],
    "factscore_labeled": sorted((DATA_ROOT / "factscore" / "labeled").glob("factscore_labeled_*.jsonl")),
    "factscore_unlabeled": sorted((DATA_ROOT / "factscore" / "unlabeled").glob("factscore_unlabeled_*.jsonl")),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest() -> dict:
    with open(OUT_ROOT / "summary_stats.json", encoding="utf-8") as f:
        summary = json.load(f)
    counts = summary["per_dataset_counts"]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_stats_used": str(OUT_ROOT / "summary_stats.json"),
        "datasets": {},
    }

    for name, paths in RAW_FILES.items():
        manifest["datasets"][name] = {
            "files": [
                {"path": str(p), "sha256": _sha256(p), "size_bytes": p.stat().st_size}
                for p in paths
            ],
            "processed_item_count": counts.get(name, 0),
        }

    return manifest


if __name__ == "__main__":
    manifest = build_manifest()
    out_path = OUT_ROOT / "dataset_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written to {out_path}")
    print(json.dumps({k: v["processed_item_count"] for k, v in manifest["datasets"].items()}, indent=2))