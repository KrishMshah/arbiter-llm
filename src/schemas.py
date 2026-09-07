"""
src/schemas.py
All Pydantic models for ARBITER live here. Consolidated by design —
one schema file, not split across modules, per team preference.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, Field # pyright: ignore[reportMissingImports]


# ============================================================================
# Benchmark schemas (Phase "2-in-doc" — done, tested against real data)
# ============================================================================

class TaskType(str, Enum):
    FACTUAL_QA = "factual_qa"
    SUMMARISATION = "summarisation"
    REASONING = "reasoning"
    CREATIVE = "creative"


class DatasetSource(str, Enum):
    CHATBOT_ARENA = "chatbot_arena"
    MT_BENCH_HUMAN = "mt_bench_human"
    MT_BENCH_GPT4 = "mt_bench_gpt4_pair"
    TRUTHFULQA = "truthfulqa"
    SUMMEVAL = "summeval"
    FACTSCORE_LABELED = "factscore_labeled"
    FACTSCORE_UNLABELED = "factscore_unlabeled"
    SELF_GENERATED = "self_generated"


class QualityScoreBasis(str, Enum):
    ELO_NORMALIZED = "elo_normalized"
    EXPERT_MEAN = "expert_annotation_mean"
    FACTSCORE_RATIO = "factscore_supported_ratio"
    BINARY_TRUTHFUL_LABEL = "binary_truthful_label"
    NONE = "none"


class BenchmarkItem(BaseModel):
    """Unified shape every dataset loader normalizes into. output_text is
    always what the critics evaluate. human_quality_score is always 1-10,
    but derived differently per dataset — see human_quality_score_basis."""
    item_id: str
    dataset_source: DatasetSource
    task_type: TaskType
    input_prompt: str
    output_text: str
    model_source: Optional[str] = None
    reference_text: Optional[list[str]] = None
    human_quality_score: Optional[float] = Field(default=None, ge=1.0, le=10.0)
    human_quality_score_basis: QualityScoreBasis = QualityScoreBasis.NONE
    ground_truth_label: Optional[str] = None
    # Ground-truth hallucinated sentences, for RQ5 validation.
    # Populated for truthfulqa (untruthful items) and factscore_labeled.
    known_hallucination_spans: Optional[list[str]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=True)


# ============================================================================
# Critic schemas
# ============================================================================

class Severity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class Issue(BaseModel):
    description: str
    quoted_evidence: str
    severity: Severity
    dimension: str  # "factual_accuracy" | "logical_consistency" | "completeness"


class CritiqueOutput(BaseModel):
    """One critic's full evaluation. All critics score all dimensions that
    routing.py selects for the task type — dimension_scores holds one entry
    per dimension actually evaluated (see dimensions_evaluated).

    A critic can fail (timeout, API error, etc. — Phase 2 concern once real
    calls exist). critic_failed=True marks this; dimension_scores/issues/
    dimensions_evaluated are empty and self_confidence is None in that case.
    Every other part of the system (disagreement detection, ML features)
    must filter these out before comparing critics against each other —
    see disagreement.py."""
    critic_id: str
    model_used: str
    dimension_scores: dict[str, int]  # e.g. {"factual_accuracy": 4, "completeness": 3}
    issues: list[Issue]
    self_confidence: Optional[int] = Field(default=None, ge=1, le=5)
    reasoning: str
    dimensions_evaluated: list[str]
    critic_failed: bool = False
    failure_reason: Optional[str] = None

    model_config = ConfigDict(use_enum_values=True)


# ============================================================================
# Routing schemas
# ============================================================================

class RoutingDecision(BaseModel):
    """Routing selects DIMENSIONS, not critics — all three critics always
    run; this just scopes which dimensions each one scores."""
    task_type: TaskType
    dimensions_selected: list[str]
    dimensions_skipped: list[str]
    routing_reason: str

    model_config = ConfigDict(use_enum_values=True)


# ============================================================================
# Disagreement schemas
# ============================================================================

class DisagreementType(str, Enum):
    SCORE_DIVERGENCE = "score_divergence"
    ISSUE_DETECTION_MISS = "issue_detection_miss"          # ground-truth confirmed: non-flaggers missed a real issue
    SEVERITY_MISMATCH = "severity_mismatch"
    FALSE_POSITIVE_FLAG = "false_positive_flag"             # ground-truth confirmed: flaggers were wrong
    ISSUE_PRESENCE_SPLIT = "issue_presence_split"           # no ground truth available, direction unresolved


class DisagreementEvent(BaseModel):
    disagreement_type: DisagreementType
    critics_involved: list[str]
    description: str
    severity: Severity

    model_config = ConfigDict(use_enum_values=True)


class DisagreementMatrix(BaseModel):
    has_disagreement: bool
    events: list[DisagreementEvent]
    max_score_gap: int
    disagreement_count: int


class MLFeatures(BaseModel):
    score_gap_max: int
    score_gap_mean: float
    score_variance: float
    score_mean: float
    disagreement_count: int
    disagreement_type_score_divergence: int
    disagreement_type_issue_miss: int
    disagreement_type_severity_mismatch: int
    disagreement_type_false_positive: int
    disagreement_type_issue_presence_split: int  # unresolved splits, no ground truth
    critic_confidence_mean: float
    critic_confidence_min: int
    issue_severity_max: int  # 0=none, 1=minor, 2=major, 3=critical
    issue_count_total: int
    critics_used_count: int
    task_type_factual_qa: int
    task_type_summarisation: int
    task_type_reasoning: int
    task_type_creative: int


# ============================================================================
# ML Arbitrator schemas
# ============================================================================

class MLArbitratorOutput(BaseModel):
    predicted_quality_score: float = Field(ge=1.0, le=10.0)
    arbitration_confidence: float = Field(ge=0.0, le=1.0)
    model_used: str  # "xgboost" | "lightgbm" | "stub"
    escalate_to_gpt: bool
    feature_importances: Optional[dict] = None


# ============================================================================
# Hallucination schemas
# ============================================================================

class OriginSentence(BaseModel):
    sentence_index: int
    text: str
    flagged_by: list[str]
    issue_description: str


class DependentSentence(BaseModel):
    sentence_index: int
    text: str
    depends_on_origin_index: int
    dependency_type: str  # "factual_reference" | "logical_extension" | "pronoun_reference"


class HallucinationTrace(BaseModel):
    has_hallucination: bool
    origin_sentences: list[OriginSentence]
    dependent_sentences: list[DependentSentence]
    propagation_depth: int
    total_affected_sentences: int


# ============================================================================
# Verdict / Adjudicator schemas
# ============================================================================

class DismissedFlag(BaseModel):
    description: str
    raised_by: str  # critic_id
    dismissal_reason: str


class Verdict(BaseModel):
    quality_score: int = Field(ge=1, le=10)
    confidence: int = Field(ge=1, le=5)
    confirmed_issues: list[Issue]
    dismissed_flags: list[DismissedFlag]
    adjudicated: bool
    adjudicated_by: str  # "ml_arbitrator" | "gpt4o_adjudicator"
    adjudicator_reasoning: Optional[str] = None


# ============================================================================
# Master pipeline result
# ============================================================================

class ArbitrationResult(BaseModel):
    input_id: str
    task_type: TaskType
    original_output: str
    routing_decision: RoutingDecision
    critiques: list[CritiqueOutput]
    disagreement_matrix: DisagreementMatrix
    ml_features: MLFeatures
    ml_arbitrator_output: MLArbitratorOutput
    verdict: Verdict
    hallucination_trace: HallucinationTrace
    cost_breakdown: dict[str, float]  # per critic_id / "adjudicator", sums to total_cost_usd
    total_cost_usd: float
    latency_ms: int

    model_config = ConfigDict(use_enum_values=True)