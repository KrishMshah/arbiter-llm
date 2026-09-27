"""
src/arbitrator.py

ML arbitration layer. Loads the real trained model (XGBoost, per
models/model_meta.json) at inference time. Falls back to a cold-start
heuristic if no model exists or fewer than COLD_START_MIN_ITEMS have
been processed.

features_to_vector()'s field order (19 features) must NOT change without
retraining -- the saved model's weights are positional.
"""

import json
import pickle
from pathlib import Path

from src.config import ML_CONFIDENCE_THRESHOLD
from src.schemas import MLFeatures, MLArbitratorOutput

SAVED_MODELS_DIR = Path("models")
MODEL_META_PATH = SAVED_MODELS_DIR / "model_meta.json"
COLD_START_MIN_ITEMS = 100


def features_to_vector(f: MLFeatures) -> list[float]:
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


_model = None
_model_meta = None
_model_load_attempted = False


def _load_trained_model():
    global _model, _model_meta, _model_load_attempted
    if _model_load_attempted:
        return _model, _model_meta
    _model_load_attempted = True

    if not MODEL_META_PATH.exists():
        return None, None
    with open(MODEL_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    model_path = Path(meta["model_path"])
    if not model_path.exists():
        return None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    _model, _model_meta = model, meta
    return _model, _model_meta


def compute_arbitration_confidence(features: MLFeatures) -> float:
    confidence = 1.0
    confidence -= min(features.disagreement_count * 0.08, 0.4)
    confidence -= min(features.score_variance * 0.05, 0.25)
    confidence -= max(0.0, (3.0 - features.critic_confidence_mean) * 0.1)
    confidence -= features.issue_severity_max * 0.05
    return max(0.0, min(1.0, round(confidence, 4)))


def _cold_start_arbitrate(features: MLFeatures) -> MLArbitratorOutput:
    has_disagreement = features.disagreement_count > 0
    fallback_score = max(1.0, min(10.0, features.score_mean * 2))
    return MLArbitratorOutput(
        predicted_quality_score=round(fallback_score, 2),
        arbitration_confidence=0.0 if has_disagreement else 0.5,
        model_used="stub",
        escalate_to_gpt=has_disagreement,
        feature_importances=None,
    )


def run_arbitrator(features: MLFeatures) -> MLArbitratorOutput:
    model, meta = _load_trained_model()
    if model is None or (meta["n_train"] + meta["n_test"]) < COLD_START_MIN_ITEMS:
        return _cold_start_arbitrate(features)

    vector = features_to_vector(features)
    predicted_score = max(1.0, min(10.0, float(model.predict([vector])[0])))
    confidence = compute_arbitration_confidence(features)

    return MLArbitratorOutput(
        predicted_quality_score=round(predicted_score, 2),
        arbitration_confidence=confidence,
        model_used=meta["winner"],
        escalate_to_gpt=confidence < ML_CONFIDENCE_THRESHOLD,
        feature_importances=meta.get("feature_importances"),
    )