"""
src/arbitrator.py

ML arbitration layer. Phase 1 = stub only — no trained model exists yet
(trainer.py and the first 200 benchmark items are Phase 2). Stub returns a
deterministic, seeded confidence/score so pipeline runs are reproducible.

Phase 2 TODO: this file must be rebuilt, not just extended. Specifically:
  - Load a real trained XGBoost/LightGBM model from src/arbitrator/saved_models/
    (currently no model file exists — run_arbitrator() never checks for one)
  - Add the cold-start guardrail from the project spec: fall back to GPT-4o
    for ALL disagreements until at least 100 benchmark items have been
    processed, or until no trained model exists yet — neither check exists
    right now, the stub is unconditionally active regardless of item count
  - Replace the stub body in run_arbitrator() with real model.predict()
    inference on the feature vector
  - features_to_vector()'s field order (19 features) must NOT change without
    retraining — Phase 2 training will depend on this exact order staying
    stable, since the saved model learns weights against these positions
"""

import hashlib
import random

from src.config import ML_CONFIDENCE_THRESHOLD
from src.schemas import MLFeatures, MLArbitratorOutput


def features_to_vector(f: MLFeatures) -> list[float]:
    """Flat feature vector, order matches MLFeatures field order exactly.
    19 features (18 from the original spec + issue_presence_split, added
    when we made disagreement direction ground-truth-aware)."""
    return [
        f.score_gap_max, f.score_gap_mean, f.score_variance, f.score_mean,
        f.disagreement_count,
        f.disagreement_type_score_divergence, f.disagreement_type_issue_miss,
        f.disagreement_type_severity_mismatch, f.disagreement_type_false_positive,
        f.disagreement_type_issue_presence_split,
        f.critic_confidence_mean, f.critic_confidence_min,
        f.issue_severity_max, f.issue_count_total, f.critics_used_count,
        f.task_type_factual_qa, f.task_type_summarisation,
        f.task_type_reasoning, f.task_type_creative,
    ]


def _seeded_rng(vector: list[float]) -> random.Random:
    seed = int(hashlib.sha256(str(vector).encode()).hexdigest(), 16)
    return random.Random(seed)


def run_arbitrator(features: MLFeatures) -> MLArbitratorOutput:
    """Phase 1 stub. Real model inference (load saved_models/*.pkl, predict)
    replaces this body in Phase 2 once trainer.py produces a trained model.
    See module-level Phase 2 TODO for the full rebuild checklist."""
    vector = features_to_vector(features)
    rng = _seeded_rng(vector)

    predicted_score = round(rng.uniform(1.0, 10.0), 2)
    confidence = round(rng.uniform(0.0, 1.0), 2)

    return MLArbitratorOutput(
        predicted_quality_score=predicted_score,
        arbitration_confidence=confidence,
        model_used="stub",
        escalate_to_gpt=confidence < ML_CONFIDENCE_THRESHOLD,
        feature_importances=None,
    )