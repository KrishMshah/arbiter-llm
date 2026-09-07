"""
src/disagreement.py

Detector + classifier + ML feature extractor. Disagreement is computed
PER DIMENSION, then pooled.

Issue presence splits (one critic flags, another doesn't) are directional
claims — "X missed something real" vs "X flagged something that isn't
there" — and we only make that claim when ground truth actually supports
it (TruthfulQA labels, FActScore hallucination spans, factual_accuracy
dimension only). Everywhere else, the split is recorded neutrally as
ISSUE_PRESENCE_SPLIT with no direction asserted — no majority-rule
guessing, since majority agreement isn't evidence of correctness in a
framework whose whole premise is that critics can share blind spots.
"""

from itertools import combinations
from statistics import mean, pvariance
from typing import Optional

from src.schemas import (
    CritiqueOutput, DisagreementMatrix, DisagreementEvent, DisagreementType,
    Severity, MLFeatures, TaskType, BenchmarkItem, DatasetSource, QualityScoreBasis,
)

_SEVERITY_RANK = {Severity.MINOR.value: 1, Severity.MAJOR.value: 2, Severity.CRITICAL.value: 3}
SCORE_GAP_THRESHOLD = 2


def _successful(critiques: list[CritiqueOutput]) -> list[CritiqueOutput]:
    """Filters out critics that failed (timeout, API error, etc.) — a failed
    critic has no real dimension_scores/issues to compare, so every
    comparison in this file operates on this filtered list, never the raw
    critiques list directly."""
    return [c for c in critiques if not c.critic_failed]


def _routed_dimensions(successful: list[CritiqueOutput]) -> list[str]:
    """All successful critics evaluate the same dimensions under the current
    design — just read them off the first one. Must be called with the
    SUCCESSFUL list, not the raw critiques list — a failed critic's
    dimensions_evaluated is always empty, and if it happened to be at index
    0, that would silently skip disagreement detection for every dimension
    even when the other critics succeeded fully."""
    return successful[0].dimensions_evaluated if successful else []


def _resolve_ground_truth_issue(benchmark_item: Optional[BenchmarkItem], dimension: str) -> Optional[bool]:
    """True = ground truth confirms an issue really exists on this dimension.
    False = ground truth confirms no issue exists.
    None = no reliable ground truth available — caller must not assert a direction.

    Only factual_accuracy currently has ground truth signal, and only for
    two dataset sources. Everything else returns None on purpose.
    """
    if benchmark_item is None or dimension != "factual_accuracy":
        return None

    if benchmark_item.dataset_source == DatasetSource.TRUTHFULQA.value:
        if benchmark_item.ground_truth_label == "untruthful":
            return True
        if benchmark_item.ground_truth_label == "truthful":
            return False
        return None

    if benchmark_item.dataset_source == DatasetSource.FACTSCORE_LABELED.value:
        if benchmark_item.human_quality_score_basis == QualityScoreBasis.FACTSCORE_RATIO.value:
            return bool(benchmark_item.known_hallucination_spans)
        return None

    return None


def detect_disagreement(
    critiques: list[CritiqueOutput],
    benchmark_item: Optional[BenchmarkItem] = None,
) -> DisagreementMatrix:
    events: list[DisagreementEvent] = []
    all_gaps: list[int] = []
    successful = _successful(critiques)

    for dim in _routed_dimensions(successful):
        dim_scores = {c.critic_id: c.dimension_scores[dim] for c in successful if dim in c.dimension_scores}
        dim_issues = {c.critic_id: [i for i in c.issues if i.dimension == dim] for c in successful}

        # Score Divergence — pairwise, per dimension. No directional claim, so no ground truth needed.
        for (id1, s1), (id2, s2) in combinations(dim_scores.items(), 2):
            gap = abs(s1 - s2)
            all_gaps.append(gap)
            if gap >= SCORE_GAP_THRESHOLD:
                events.append(DisagreementEvent(
                    disagreement_type=DisagreementType.SCORE_DIVERGENCE,
                    critics_involved=[id1, id2],
                    description=f"On {dim}: {id1} scored {s1}, {id2} scored {s2}",
                    severity=Severity.MAJOR if gap >= 3 else Severity.MINOR,
                ))

        # Issue presence split — directional label ONLY if ground truth resolves it
        flaggers = [cid for cid, issues in dim_issues.items() if issues]
        non_flaggers = [cid for cid, issues in dim_issues.items() if not issues]

        if flaggers and non_flaggers:
            gt = _resolve_ground_truth_issue(benchmark_item, dim)

            if gt is True:
                events.append(DisagreementEvent(
                    disagreement_type=DisagreementType.ISSUE_DETECTION_MISS,
                    critics_involved=non_flaggers,
                    description=f"On {dim}: ground truth confirms a real issue; {non_flaggers} missed it",
                    severity=Severity.MAJOR,
                ))
            elif gt is False:
                events.append(DisagreementEvent(
                    disagreement_type=DisagreementType.FALSE_POSITIVE_FLAG,
                    critics_involved=flaggers,
                    description=f"On {dim}: ground truth confirms no issue exists; {flaggers} flagged incorrectly",
                    severity=Severity.MAJOR,
                ))
            else:
                events.append(DisagreementEvent(
                    disagreement_type=DisagreementType.ISSUE_PRESENCE_SPLIT,
                    critics_involved=flaggers + non_flaggers,
                    description=f"On {dim}: {flaggers} flagged an issue, {non_flaggers} did not — no ground truth to resolve direction",
                    severity=Severity.MINOR,
                ))

        # Severity Mismatch — among critics that DID flag, do severities differ?
        # Not a directional claim (nobody's "right"), so no ground truth needed.
        if len(flaggers) >= 2:
            max_sev_by_critic = {
                cid: max(dim_issues[cid], key=lambda i: _SEVERITY_RANK[i.severity]).severity
                for cid in flaggers
            }
            if len(set(max_sev_by_critic.values())) > 1:
                events.append(DisagreementEvent(
                    disagreement_type=DisagreementType.SEVERITY_MISMATCH,
                    critics_involved=flaggers,
                    description=f"On {dim}: critics flagged issues but rated severity differently",
                    severity=Severity.MAJOR,
                ))

    max_gap = max(all_gaps) if all_gaps else 0
    return DisagreementMatrix(
        has_disagreement=len(events) > 0,
        events=events,
        max_score_gap=max_gap,
        disagreement_count=len(events),
    )


def extract_ml_features(
    critiques: list[CritiqueOutput],
    matrix: DisagreementMatrix,
    task_type: TaskType | str,
) -> MLFeatures:
    successful = _successful(critiques)
    all_scores = [s for c in successful for s in c.dimension_scores.values()]
    confidences = [c.self_confidence for c in successful]  # never None here — only successful critics have a real value
    all_issues = [i for c in successful for i in c.issues]
    task_key = task_type.value if isinstance(task_type, TaskType) else task_type

    type_counts = {t: 0 for t in DisagreementType}
    for e in matrix.events:
        type_counts[DisagreementType(e.disagreement_type)] += 1

    max_severity_rank = max((_SEVERITY_RANK[i.severity] for i in all_issues), default=0)

    return MLFeatures(
        score_gap_max=matrix.max_score_gap,
        score_gap_mean=mean(abs(a - b) for a, b in combinations(all_scores, 2)) if len(all_scores) > 1 else 0.0,
        score_variance=pvariance(all_scores) if len(all_scores) > 1 else 0.0,
        score_mean=mean(all_scores) if all_scores else 0.0,
        disagreement_count=matrix.disagreement_count,
        disagreement_type_score_divergence=type_counts[DisagreementType.SCORE_DIVERGENCE],
        disagreement_type_issue_miss=type_counts[DisagreementType.ISSUE_DETECTION_MISS],
        disagreement_type_severity_mismatch=type_counts[DisagreementType.SEVERITY_MISMATCH],
        disagreement_type_false_positive=type_counts[DisagreementType.FALSE_POSITIVE_FLAG],
        disagreement_type_issue_presence_split=type_counts[DisagreementType.ISSUE_PRESENCE_SPLIT],
        critic_confidence_mean=mean(confidences) if confidences else 0.0,
        critic_confidence_min=min(confidences) if confidences else 0,
        issue_severity_max=max_severity_rank,
        issue_count_total=len(all_issues),
        critics_used_count=len(successful),  # successful critics only — a failed critic didn't "participate"
        task_type_factual_qa=int(task_key == TaskType.FACTUAL_QA.value),
        task_type_summarisation=int(task_key == TaskType.SUMMARISATION.value),
        task_type_reasoning=int(task_key == TaskType.REASONING.value),
        task_type_creative=int(task_key == TaskType.CREATIVE.value),
    )