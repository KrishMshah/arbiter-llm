"""
scripts/sample_arena.py

Samples 130 complete pairs (260 items) from Chatbot Arena for the
Zheng et al. agreement-rate comparison. Separate from sampling.py's
4-dataset training sample. Fixed-seed, deterministic.

Confirmed against benchmark.py: metadata["pair_id"] is always set for
Arena items, and incomplete pairs (one side missing) are possible since
load_chatbot_arena() skips a side with no assistant reply.
"""

import json
import random
from pathlib import Path
from typing import Optional

OUT_ROOT = Path("results/processed")
ARENA_PATH = OUT_ROOT / "chatbot_arena.jsonl"
SEED = 42
N_PAIRS = 130


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _pair_id_for(item: dict) -> Optional[str]:
    # metadata["pair_id"] is always set by benchmark.py -- fallback below
    # is just a safety net, not the expected path.
    meta_pair_id = item.get("metadata", {}).get("pair_id")
    if meta_pair_id:
        return meta_pair_id
    item_id = item.get("item_id", "")
    if item_id.endswith("_a") or item_id.endswith("_b"):
        return item_id[:-2]
    return None


def group_into_pairs(items: list[dict]) -> dict[str, dict[str, dict]]:
    # Only keep pairs with both sides -- one side can be missing if that
    # model's turn had no assistant reply (benchmark.py skips those).
    pairs: dict[str, dict[str, dict]] = {}
    unmatched = 0
    for item in items:
        pid = _pair_id_for(item)
        if pid is None:
            unmatched += 1
            continue
        side = "a" if item["item_id"].endswith("_a") else "b"
        pairs.setdefault(pid, {})[side] = item

    complete = {pid: sides for pid, sides in pairs.items() if "a" in sides and "b" in sides}
    incomplete = len(pairs) - len(complete)

    print(f"Loaded {len(items)} Arena items -> {len(pairs)} pair_ids "
          f"({len(complete)} complete, {incomplete} incomplete, {unmatched} unmatched)")
    return complete


def build_arena_sample():
    items = _load_jsonl(ARENA_PATH)
    complete_pairs = group_into_pairs(items)

    if len(complete_pairs) < N_PAIRS:
        print(f"WARNING: only {len(complete_pairs)} complete pairs available, wanted {N_PAIRS}")

    rng = random.Random(SEED)
    sampled_ids = rng.sample(sorted(complete_pairs.keys()), min(N_PAIRS, len(complete_pairs)))

    flat_items = []
    pair_meta = {}
    for pid in sampled_ids:
        sides = complete_pairs[pid]
        flat_items.append(sides["a"])
        flat_items.append(sides["b"])
        # outcome (win/loss/tie/tie_bothbad) lives inside raw_metadata --
        # that's the actual human vote the agreement-rate script needs later.
        pair_meta[pid] = {
            "item_id_a": sides["a"]["item_id"],
            "item_id_b": sides["b"]["item_id"],
            "raw_metadata_a": sides["a"].get("metadata", {}),
            "raw_metadata_b": sides["b"].get("metadata", {}),
        }

    rng.shuffle(flat_items)
    return flat_items, pair_meta


if __name__ == "__main__":
    flat_items, pair_meta = build_arena_sample()

    out_path = OUT_ROOT / "run_sample_arena.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in flat_items:
            f.write(json.dumps(item) + "\n")

    meta_path = OUT_ROOT / "run_sample_arena_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "n_pairs": len(pair_meta),
            "n_items": len(flat_items),
            "pairs": pair_meta,
        }, f, indent=2)

    print(f"\nSampled {len(pair_meta)} pairs ({len(flat_items)} items) -> {out_path}")
    print(f"Pair metadata -> {meta_path}")