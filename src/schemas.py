"""
src/schemas.py

All Pydantic models for ARBITER live here. Only benchmark-related schemas
exist so far (critics/routing/disagreement/verdict schemas get added here
in later phases, once those modules are actually built).
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field  # pyright: ignore[reportMissingImports]


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

    class Config:
        use_enum_values = True