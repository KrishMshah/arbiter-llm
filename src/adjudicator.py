"""
src/adjudicator.py

GPT-4o adjudicator. Only triggered when arbitrator.py's escalate_to_gpt is
True (arbitration_confidence < ML_CONFIDENCE_THRESHOLD). Phase 1 = stub,
not wired into a real pipeline call yet — produces a valid Verdict so
end-to-end MOCK_MODE runs complete, using the guardrail-spec fallback
behavior (majority vote on score + union of confirmed issues) as the stub
logic itself, since that's what a failed real call falls back to anyway.

Phase 2 TODO: real GPT-4o call, ChromaDB evidence retrieval (retrieved_evidence
is always [] here), 3000-token context cap enforcement, and real reasoning
to populate dismissed_flags (currently always empty — no reasoning to
decide what to dismiss without a real model).
"""

import hashlib
import random

from src.schemas import CritiqueOutput, DisagreementMatrix, MLArbitratorOutput, Verdict, Issue


def _seeded_rng(seed_input: str) -> random.Random:
    seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16)
    return random.Random(seed)


def _rescale_1_5_to_1_10(score_1_5: float) -> float:
    """Same rescale formula benchmark.py uses for SummEval — keeps the
    1-5 critic scale and 1-10 verdict scale consistent project-wide."""
    return 1.0 + (score_1_5 - 1.0) * (9.0 / 4.0)


def retrieve_evidence(original_output: str) -> list[str]:
    """Phase 1 stub — no ChromaDB set up yet. Always returns empty."""
    return []


def run_adjudicator(
    original_output: str,
    critiques: list[CritiqueOutput],
    disagreement_matrix: DisagreementMatrix,
    ml_output: MLArbitratorOutput,
) -> Verdict:
    all_dim_scores = [s for c in critiques for s in c.dimension_scores.values()]
    all_issues: list[Issue] = [i for c in critiques for i in c.issues]

    mean_score_1_5 = sum(all_dim_scores) / len(all_dim_scores) if all_dim_scores else 3.0
    quality_score_1_10 = round(_rescale_1_5_to_1_10(mean_score_1_5))
    quality_score_1_10 = max(1, min(10, quality_score_1_10))  # clamp per guardrails

    rng = _seeded_rng(original_output)
    confidence = rng.randint(1, 5)

    return Verdict(
        quality_score=quality_score_1_10,
        confidence=confidence,
        confirmed_issues=all_issues,
        dismissed_flags=[],  # Phase 2 TODO — needs real reasoning to decide what to dismiss
        adjudicated=True,
        adjudicated_by="gpt4o_adjudicator",
        adjudicator_reasoning=(
            "[MOCK] Stub adjudicator: quality_score is the mean of all critic "
            "dimension scores rescaled 1-10, confirmed_issues is the union of "
            "all flagged issues (no filtering). No real reasoning applied."
        ),
    )