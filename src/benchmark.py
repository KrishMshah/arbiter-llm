"""
src/benchmark.py

Loads all 5 downloaded datasets, normalizes each into BenchmarkItem, writes
results/processed/*.jsonl. Run with: python -m src.benchmark

Key decisions (kept brief — ask if you want the full reasoning on any of these):

- Chatbot Arena & MT-Bench are pairwise (A vs B + winner). Split into 2
  single-output items each; quality score = that model's Elo rating,
  rescaled 1-10. This is a MODEL-level proxy, not an instance-level human
  judgment — every output from a model gets that model's score.
- Chatbot Arena has no task-type metadata -> classified via keyword regex.
- MT-Bench has real per-question categories -> hardcoded mapping below
  (sourced from lm-sys/FastChat), folded into ARBITER's 4 task types.
- TruthfulQA has no model outputs, just an answer key -> best_answer becomes
  a "truthful" item (score 9.0), first incorrect_answer becomes an
  "untruthful" item (score 2.0). Real ground truth, zero generation cost.
- SummEval quality score = mean of 3 expert annotators across 4 dimensions,
  rescaled 1-10. Reference summaries only (no source articles - confirmed).
- FActScore labeled: quality score = supported/(supported+not_supported)
  atomic facts, rescaled 1-10. Sentences with a not-supported fact become
  known_hallucination_spans - real ground truth for RQ5 tracer validation.
  FActScore unlabeled: no human labels, loaded for pipeline volume only.
"""

from __future__ import annotations
import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean

import pandas as pd

from src.schemas import BenchmarkItem, DatasetSource, TaskType, QualityScoreBasis

DATA_ROOT = Path("benchmark_data")
OUT_ROOT = Path("results/processed")


# ============================================================================
# Elo rating (shared by chatbot_arena and mt_bench loaders)
# ============================================================================

_K_FACTOR = 32
_INITIAL_RATING = 1000.0


def _expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))


def _compute_elo(matches: list[tuple[str, str, float]]) -> dict[str, float]:
    """matches: (model_a, model_b, score_a) where score_a is 1/0.5/0."""
    ratings: dict[str, float] = defaultdict(lambda: _INITIAL_RATING)
    for a, b, score_a in matches:
        ra, rb = ratings[a], ratings[b]
        exp_a = _expected_score(ra, rb)
        ratings[a] = ra + _K_FACTOR * (score_a - exp_a)
        ratings[b] = rb + _K_FACTOR * ((1 - score_a) - (1 - exp_a))
    return dict(ratings)


def _normalize_elo(ratings: dict[str, float]) -> dict[str, float]:
    if not ratings:
        return {}
    vals = list(ratings.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 5.5 for k in ratings}
    return {k: 1.0 + 9.0 * (v - lo) / (hi - lo) for k, v in ratings.items()}


def _outcome_to_score_a(winner: str) -> float | None:
    return {"model_a": 1.0, "model_b": 0.0, "tie": 0.5}.get(winner)  # "tie (bothbad)" -> None, excluded


def _last_assistant_msg(conv) -> str | None:
    for turn in reversed(list(conv)):
        if turn.get("role") == "assistant":
            return turn.get("content")
    return None


def _first_user_msg(conv) -> str:
    for turn in conv:
        if turn.get("role") == "user":
            return turn.get("content", "")
    return ""


# ============================================================================
# Chatbot Arena
# ============================================================================

_SUMMARY_RE = re.compile(r"\b(summarize|summarise|summary|tl;?dr|shorten this|condense)\b", re.I)
_FACTUAL_RE = re.compile(
    r"^(what is|what are|who is|who was|when did|when was|where is|"
    r"define|explain the difference|how many|how much|what year)\b", re.I
)
_REASONING_RE = re.compile(
    r"\b(solve|calculate|prove|algorithm|debug|write a function|write code|"
    r"python|javascript|java |sql|regex|step by step|logic puzzle|math problem)\b", re.I
)


def _classify_task_type(prompt: str) -> TaskType:
    prompt = prompt.strip()
    if _SUMMARY_RE.search(prompt):
        return TaskType.SUMMARISATION
    if _FACTUAL_RE.search(prompt[:80]):
        return TaskType.FACTUAL_QA
    if _REASONING_RE.search(prompt):
        return TaskType.REASONING
    return TaskType.CREATIVE


def load_chatbot_arena(parquet_path: str | Path) -> list[BenchmarkItem]:
    df = pd.read_parquet(parquet_path).sort_values("tstamp").reset_index(drop=True)

    matches = [
        (row["model_a"], row["model_b"], _outcome_to_score_a(row["winner"]))
        for _, row in df.iterrows()
    ]
    matches = [(a, b, s) for a, b, s in matches if s is not None]
    quality_scores = _normalize_elo(_compute_elo(matches))

    items = []
    for idx, row in df.iterrows():
        pair_id = f"arena_{idx:06d}"
        conv_a, conv_b = row["conversation_a"], row["conversation_b"]
        prompt = _first_user_msg(conv_a)
        task_type = _classify_task_type(prompt)

        for side, conv, model, opponent in (
            ("a", conv_a, row["model_a"], row["model_b"]),
            ("b", conv_b, row["model_b"], row["model_a"]),
        ):
            output = _last_assistant_msg(conv)
            if not output:
                continue
            outcome = (
                "win" if row["winner"] == f"model_{side}"
                else "loss" if row["winner"] in ("model_a", "model_b")
                else "tie" if row["winner"] == "tie" else "tie_bothbad"
            )
            items.append(BenchmarkItem(
                item_id=f"{pair_id}_{side}",
                dataset_source=DatasetSource.CHATBOT_ARENA,
                task_type=task_type,
                input_prompt=prompt,
                output_text=output,
                model_source=model,
                human_quality_score=quality_scores.get(model),
                human_quality_score_basis=QualityScoreBasis.ELO_NORMALIZED,
                metadata={
                    "pair_id": pair_id, "opponent_model": opponent, "outcome": outcome,
                    "turn_count": int(row["turn"]), "full_conversation": list(conv),
                },
            ))

    print(f"[chatbot_arena] {len(df)} pairs -> {len(items)} items, {len(quality_scores)} models rated")
    return items


# ============================================================================
# MT-Bench
# ============================================================================

# question_id -> category, sourced from lm-sys/FastChat's official 80-question set
_MT_BENCH_CATEGORIES = {
    **{qid: "writing" for qid in range(81, 91)},
    **{qid: "roleplay" for qid in range(91, 101)},
    **{qid: "reasoning" for qid in range(101, 111)},
    **{qid: "math" for qid in range(111, 121)},
    **{qid: "coding" for qid in range(121, 131)},
    **{qid: "extraction" for qid in range(131, 141)},
    **{qid: "stem" for qid in range(141, 151)},
    **{qid: "humanities" for qid in range(151, 161)},
}

# MT-Bench's 8 categories folded into ARBITER's 4 routing task types
_CATEGORY_TO_TASK_TYPE = {
    "writing": TaskType.CREATIVE, "roleplay": TaskType.CREATIVE,
    "reasoning": TaskType.REASONING, "math": TaskType.REASONING, "coding": TaskType.REASONING,
    "extraction": TaskType.FACTUAL_QA, "stem": TaskType.FACTUAL_QA, "humanities": TaskType.FACTUAL_QA,
}


def _load_mt_bench_file(parquet_path, dataset_source: DatasetSource, id_prefix: str) -> list[BenchmarkItem]:
    df = pd.read_parquet(parquet_path)

    matches = [
        (row["model_a"], row["model_b"], _outcome_to_score_a(row["winner"]))
        for _, row in df.iterrows()
    ]
    matches = [(a, b, s) for a, b, s in matches if s is not None]
    quality_scores = _normalize_elo(_compute_elo(matches))

    items = []
    for idx, row in df.iterrows():
        pair_id = f"{id_prefix}_{idx:06d}"
        qid = int(row["question_id"])
        raw_category = _MT_BENCH_CATEGORIES.get(qid)
        task_type = _CATEGORY_TO_TASK_TYPE.get(raw_category, TaskType.REASONING)

        conv_a, conv_b = row["conversation_a"], row["conversation_b"]
        prompt = _first_user_msg(conv_a)

        for side, conv, model, opponent in (
            ("a", conv_a, row["model_a"], row["model_b"]),
            ("b", conv_b, row["model_b"], row["model_a"]),
        ):
            output = _last_assistant_msg(conv)
            if not output:
                continue
            outcome = (
                "win" if row["winner"] == f"model_{side}"
                else "loss" if row["winner"] in ("model_a", "model_b") else "tie"
            )
            items.append(BenchmarkItem(
                item_id=f"{pair_id}_{side}",
                dataset_source=dataset_source,
                task_type=task_type,
                input_prompt=prompt,
                output_text=output,
                model_source=model,
                human_quality_score=quality_scores.get(model),
                human_quality_score_basis=QualityScoreBasis.ELO_NORMALIZED,
                metadata={
                    "pair_id": pair_id, "question_id": qid, "mt_bench_category": raw_category,
                    "opponent_model": opponent, "outcome": outcome,
                    "turn_count": int(row["turn"]), "full_conversation": list(conv),
                },
            ))

    print(f"[{id_prefix}] {len(df)} pairs -> {len(items)} items, {len(quality_scores)} models rated")
    return items


def load_mt_bench(human_parquet_path, gpt4_pair_parquet_path) -> tuple[list[BenchmarkItem], list[BenchmarkItem]]:
    """Returns (human_items, gpt4_pair_items) - kept separate, human is real
    ground truth for RQ2, gpt4_pair is a model judgment, not human."""
    human_items = _load_mt_bench_file(human_parquet_path, DatasetSource.MT_BENCH_HUMAN, "mtbench_human")
    gpt4_items = _load_mt_bench_file(gpt4_pair_parquet_path, DatasetSource.MT_BENCH_GPT4, "mtbench_gpt4")
    return human_items, gpt4_items


# ============================================================================
# TruthfulQA
# ============================================================================

_TRUTHFUL_SCORE = 9.0
_UNTRUTHFUL_SCORE = 2.0


def load_truthfulqa(parquet_path: str | Path) -> list[BenchmarkItem]:
    df = pd.read_parquet(parquet_path)
    items = []

    for idx, row in df.iterrows():
        question = row["question"]
        best_answer = row["best_answer"]
        correct = list(row["correct_answers"]) if row["correct_answers"] is not None else []
        incorrect = list(row["incorrect_answers"]) if row["incorrect_answers"] is not None else []
        meta = {"truthfulqa_type": row.get("type"), "truthfulqa_category": row.get("category")}

        items.append(BenchmarkItem(
            item_id=f"truthfulqa_{idx:04d}_truthful",
            dataset_source=DatasetSource.TRUTHFULQA,
            task_type=TaskType.FACTUAL_QA,
            input_prompt=question,
            output_text=best_answer,
            model_source="truthfulqa_human_curated",
            reference_text=correct or [best_answer],
            human_quality_score=_TRUTHFUL_SCORE,
            human_quality_score_basis=QualityScoreBasis.BINARY_TRUTHFUL_LABEL,
            ground_truth_label="truthful",
            metadata=meta,
        ))

        if incorrect:
            items.append(BenchmarkItem(
                item_id=f"truthfulqa_{idx:04d}_untruthful",
                dataset_source=DatasetSource.TRUTHFULQA,
                task_type=TaskType.FACTUAL_QA,
                input_prompt=question,
                output_text=incorrect[0],
                model_source="truthfulqa_human_curated",
                reference_text=correct or [best_answer],
                human_quality_score=_UNTRUTHFUL_SCORE,
                human_quality_score_basis=QualityScoreBasis.BINARY_TRUTHFUL_LABEL,
                ground_truth_label="untruthful",
                known_hallucination_spans=[incorrect[0]],
                metadata=meta,
            ))

    print(f"[truthfulqa] {len(df)} questions -> {len(items)} items")
    return items


# ============================================================================
# SummEval
# ============================================================================

_DIMENSIONS = ("coherence", "consistency", "fluency", "relevance")


def _dimension_mean(annotations: list[dict]) -> float | None:
    scores = [ann[d] for ann in annotations for d in _DIMENSIONS if d in ann]
    return mean(scores) if scores else None


def load_summeval(jsonl_path: str | Path) -> list[BenchmarkItem]:
    items = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            row = json.loads(line)
            expert_mean = _dimension_mean(row.get("expert_annotations", []))
            turker_mean = _dimension_mean(row.get("turker_annotations", []))
            quality_1_5 = expert_mean if expert_mean is not None else turker_mean
            quality_1_10 = 1.0 + (quality_1_5 - 1.0) * (9.0 / 4.0) if quality_1_5 is not None else None

            items.append(BenchmarkItem(
                item_id=f"summeval_{line_no:04d}_{row['id']}",
                dataset_source=DatasetSource.SUMMEVAL,
                task_type=TaskType.SUMMARISATION,
                input_prompt="Summarize the following article.",
                output_text=row["decoded"],
                model_source=row.get("model_id"),
                reference_text=row.get("references"),
                human_quality_score=quality_1_10,
                human_quality_score_basis=QualityScoreBasis.EXPERT_MEAN,
                metadata={
                    "summeval_id": row["id"], "model_id": row.get("model_id"),
                    "used_turker_fallback": expert_mean is None,
                },
            ))

    print(f"[summeval] {len(items)} items loaded")
    return items


# ============================================================================
# FActScore
# ============================================================================

def _factscore_ratio_and_spans(annotations: list[dict] | None) -> tuple[float | None, list[str]]:
    if not annotations:
        return None, []
    supported = not_supported = 0
    spans = []
    for ann in annotations:
        facts = ann.get("human-atomic-facts") or []
        has_ns = False
        for fact in facts:
            if fact.get("label") == "S":
                supported += 1
            elif fact.get("label") == "NS":
                not_supported += 1
                has_ns = True
        if has_ns:
            spans.append(ann.get("text", ""))
    total = supported + not_supported
    ratio = supported / total if total > 0 else None
    return ratio, spans


def load_factscore_labeled(jsonl_path: str | Path, model_name: str) -> list[BenchmarkItem]:
    items = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            row = json.loads(line)
            ratio, spans = _factscore_ratio_and_spans(row.get("annotations"))
            items.append(BenchmarkItem(
                item_id=f"factscore_labeled_{model_name}_{line_no:04d}",
                dataset_source=DatasetSource.FACTSCORE_LABELED,
                task_type=TaskType.FACTUAL_QA,
                input_prompt=row["input"],
                output_text=row["output"],
                model_source=model_name,
                human_quality_score=1.0 + ratio * 9.0 if ratio is not None else None,
                human_quality_score_basis=QualityScoreBasis.FACTSCORE_RATIO if ratio is not None else QualityScoreBasis.NONE,
                known_hallucination_spans=spans or None,
                metadata={"topic": row.get("topic"), "factscore_ratio_raw": ratio},
            ))
    print(f"[factscore_labeled:{model_name}] {len(items)} items")
    return items


def load_factscore_unlabeled(jsonl_path: str | Path, model_name: str) -> list[BenchmarkItem]:
    items = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            row = json.loads(line)
            items.append(BenchmarkItem(
                item_id=f"factscore_unlabeled_{model_name}_{line_no:04d}",
                dataset_source=DatasetSource.FACTSCORE_UNLABELED,
                task_type=TaskType.FACTUAL_QA,
                input_prompt=row["input"],
                output_text=row["output"],
                model_source=model_name,
                metadata={"topic": row.get("topic")},
            ))
    print(f"[factscore_unlabeled:{model_name}] {len(items)} items (no ground truth)")
    return items


# ============================================================================
# Runner
# ============================================================================

def _write_jsonl(items: list[BenchmarkItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


def build_all() -> dict:
    per_dataset: dict[str, list[BenchmarkItem]] = {}

    per_dataset["chatbot_arena"] = load_chatbot_arena(
        DATA_ROOT / "chatbot_arena" / "chatbot_arena_train.parquet"
    )

    human, gpt4 = load_mt_bench(
        DATA_ROOT / "mt_bench" / "mt_bench_human.parquet",
        DATA_ROOT / "mt_bench" / "mt_bench_gpt4_pair.parquet",
    )
    per_dataset["mt_bench_human"] = human
    per_dataset["mt_bench_gpt4_pair"] = gpt4

    per_dataset["truthfulqa"] = load_truthfulqa(
        DATA_ROOT / "truthfulqa" / "truthfulqa_validation.parquet"
    )

    per_dataset["summeval"] = load_summeval(
        DATA_ROOT / "summeval" / "summeval_model_annotations.jsonl"
    )

    labeled_items = []
    for f in sorted((DATA_ROOT / "factscore" / "labeled").glob("factscore_labeled_*.jsonl")):
        model_name = f.stem.replace("factscore_labeled_", "")
        labeled_items += load_factscore_labeled(f, model_name)
    per_dataset["factscore_labeled"] = labeled_items

    unlabeled_items = []
    for f in sorted((DATA_ROOT / "factscore" / "unlabeled").glob("factscore_unlabeled_*.jsonl")):
        model_name = f.stem.replace("factscore_unlabeled_", "")
        unlabeled_items += load_factscore_unlabeled(f, model_name)
    per_dataset["factscore_unlabeled"] = unlabeled_items

    all_items = []
    for name, items in per_dataset.items():
        _write_jsonl(items, OUT_ROOT / f"{name}.jsonl")
        all_items.extend(items)
    _write_jsonl(all_items, OUT_ROOT / "all_items.jsonl")

    summary = {
        "total_items": len(all_items),
        "per_dataset_counts": dict(Counter(i.dataset_source for i in all_items)),
        "task_type_distribution": dict(Counter(i.task_type for i in all_items)),
        "items_with_quality_score": sum(1 for i in all_items if i.human_quality_score is not None),
        "items_with_ground_truth_label": sum(1 for i in all_items if i.ground_truth_label is not None),
        "items_with_hallucination_span_ground_truth": sum(1 for i in all_items if i.known_hallucination_spans),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build_all()
