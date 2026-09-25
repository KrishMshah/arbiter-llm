"""
src/validate_run.py

Data-quality QA on an already-collected critic run. Pure local analysis --
no network calls. Ground-truth alignment (4th check) reads the sample
files directly (run_sample.jsonl + run_sample_expansion.jsonl) rather than
the raw per-dataset source files -- those two already contain exactly the
779 items in our combined results, full BenchmarkItem shape included.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path("results/processed/run_results_combined_costed.jsonl")
BENCHMARK_ITEM_PATHS = [
    Path("results/processed/run_sample.jsonl"),
    Path("results/processed/run_sample_expansion.jsonl"),
]
CRITIC_IDS = ["critic_a", "critic_b", "critic_c"]
LOW_VARIANCE_THRESHOLD = 0.3  # dimension_scores variance below this = not discriminating


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _load_results(path: Path = RESULTS_PATH) -> list[dict]:
    return _load_jsonl(path)


def failure_rate_audit(results: list[dict]) -> dict[str, dict]:
    """Per critic: how often did evaluate() throw and fall back to make_failed_critique()?"""
    counts = {cid: {"total": 0, "failed": 0} for cid in CRITIC_IDS}

    for r in results:
        for critique in r["critiques"]:
            cid = critique["critic_id"]
            counts[cid]["total"] += 1
            if critique.get("critic_failed"):
                counts[cid]["failed"] += 1

    return {
        cid: {
            "total": c["total"],
            "failed": c["failed"],
            "failure_rate": round(c["failed"] / c["total"], 4) if c["total"] else 0.0,
        }
        for cid, c in counts.items()
    }


def score_variance_audit(results: list[dict]) -> dict[str, dict]:
    """Per critic: variance of dimension_scores across the run. Near-zero
    variance means the critic isn't discriminating -- dead weight as a
    training feature, worth knowing before we build features around it."""
    scores_by_critic: dict[str, list[int]] = defaultdict(list)

    for r in results:
        for critique in r["critiques"]:
            if critique.get("critic_failed"):
                continue
            cid = critique["critic_id"]
            scores_by_critic[cid].extend(critique.get("dimension_scores", {}).values())

    out = {}
    for cid in CRITIC_IDS:
        scores = scores_by_critic.get(cid, [])
        if len(scores) < 2:
            out[cid] = {"n": len(scores), "mean": None, "variance": None, "flag": "insufficient data"}
            continue
        mean = statistics.mean(scores)
        var = statistics.variance(scores)
        out[cid] = {
            "n": len(scores),
            "mean": round(mean, 3),
            "variance": round(var, 3),
            "flag": "LOW VARIANCE -- not discriminating" if var < LOW_VARIANCE_THRESHOLD else "ok",
        }
    return out


def _normalize_for_match(text: str) -> str:
    """Strip one layer of wrapping quote marks and collapse all whitespace
    (including newlines) to single spaces, so formatting differences don't
    masquerade as hallucinated evidence."""
    import re
    text = text.strip()
    quote_pairs = [('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019")]
    for left, right in quote_pairs:
        if len(text) >= 2 and text[0] == left and text[-1] == right:
            text = text[1:-1].strip()
            break
    return re.sub(r"\s+", " ", text)


def quoted_evidence_groundedness(results: list[dict]) -> dict[str, dict]:
    """Per critic: what fraction of issues' quoted_evidence actually appears
    in the text being evaluated -- checked both raw (exact substring) and
    normalized (quote-stripped, whitespace-collapsed)."""
    counts = {cid: {"total_issues": 0, "grounded_raw": 0, "grounded_normalized": 0, "ungrounded_examples": []} for cid in CRITIC_IDS}

    for r in results:
        source_text = r.get("original_output", "")
        source_norm = _normalize_for_match(source_text)
        for critique in r["critiques"]:
            if critique.get("critic_failed"):
                continue
            cid = critique["critic_id"]
            for issue in critique.get("issues", []):
                quote = issue.get("quoted_evidence", "")
                quote_norm = _normalize_for_match(quote)
                counts[cid]["total_issues"] += 1

                if quote and quote in source_text:
                    counts[cid]["grounded_raw"] += 1
                if quote_norm and quote_norm in source_norm:
                    counts[cid]["grounded_normalized"] += 1
                elif len(counts[cid]["ungrounded_examples"]) < 3:
                    counts[cid]["ungrounded_examples"].append({
                        "input_id": r.get("input_id"),
                        "quoted_evidence": quote,
                    })

    return {
        cid: {
            "total_issues": c["total_issues"],
            "groundedness_rate_raw": round(c["grounded_raw"] / c["total_issues"], 4) if c["total_issues"] else None,
            "groundedness_rate_normalized": round(c["grounded_normalized"] / c["total_issues"], 4) if c["total_issues"] else None,
            "sample_still_ungrounded": c["ungrounded_examples"],
        }
        for cid, c in counts.items()
    }


def _load_benchmark_items(paths: list[Path]) -> dict[str, dict]:
    """item_id -> raw BenchmarkItem dict. Sourced from the sample files,
    not the full per-dataset files -- these already are exactly the 779
    items in our results, nothing to filter down."""
    items = {}
    for path in paths:
        for item in _load_jsonl(path):
            items[item["item_id"]] = item
    return items


def _pearson(xs: list[float], ys: list[float]):
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x * var_y) ** 0.5


def ground_truth_alignment(results: list[dict], benchmark_items: dict[str, dict]) -> dict:
    """Per critic:
    1. hallucination_recall -- on items where known_hallucination_spans
       confirms a real hallucination (TruthfulQA-untruthful, FActScore-
       labeled only, per schemas.py), did the critic flag a
       factual_accuracy issue?
    2. score_correlation -- critic's mean dimension_score (x2, to 1-10)
       vs human_quality_score, wherever present, any dataset.
    """
    hallu_hits = {cid: {"n": 0, "flagged": 0} for cid in CRITIC_IDS}
    score_pairs: dict[str, tuple] = {cid: ([], []) for cid in CRITIC_IDS}

    matched = 0
    for r in results:
        bi = benchmark_items.get(r.get("input_id"))
        if bi is None:
            continue
        matched += 1

        known_spans = bi.get("known_hallucination_spans")
        human_score = bi.get("human_quality_score")

        for critique in r["critiques"]:
            if critique.get("critic_failed"):
                continue
            cid = critique["critic_id"]

            if known_spans:  # non-empty list -- ground truth confirms a real hallucination
                hallu_hits[cid]["n"] += 1
                flagged = any(i.get("dimension") == "factual_accuracy" for i in critique.get("issues", []))
                if flagged:
                    hallu_hits[cid]["flagged"] += 1

            if human_score is not None:
                dims = critique.get("dimension_scores", {})
                if dims:
                    mean_critic_score = sum(dims.values()) / len(dims) * 2
                    xs, ys = score_pairs[cid]
                    xs.append(mean_critic_score)
                    ys.append(human_score)

    print(f"  matched {matched}/{len(results)} results to a benchmark item")

    out = {}
    for cid in CRITIC_IDS:
        h = hallu_hits[cid]
        xs, ys = score_pairs[cid]
        r_val = _pearson(xs, ys)
        out[cid] = {
            "hallucination_recall": {
                "n_known_hallucinated_items": h["n"],
                "flagged": h["flagged"],
                "recall": round(h["flagged"] / h["n"], 4) if h["n"] else None,
            },
            "score_correlation": {
                "n_items_with_human_score": len(xs),
                "pearson_r": round(r_val, 4) if r_val is not None else None,
            },
        }
    return out


def run_all(path: Path = RESULTS_PATH) -> dict:
    results = _load_results(path)
    benchmark_items = _load_benchmark_items(BENCHMARK_ITEM_PATHS)
    return {
        "n_items": len(results),
        "failure_rate": failure_rate_audit(results),
        "score_variance": score_variance_audit(results),
        "quoted_evidence_groundedness": quoted_evidence_groundedness(results),
        "ground_truth_alignment": ground_truth_alignment(results, benchmark_items),
    }


if __name__ == "__main__":
    report = run_all()
    print(json.dumps(report, indent=2))