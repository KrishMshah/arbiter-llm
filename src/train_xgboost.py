"""
src/train_xgboost.py 
Run: python -m src.train_xgboost
"""

import json
import pickle
from pathlib import Path

print("Starting train_xgboost.py...", flush=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

from src.training_data import build_training_data, FEATURE_NAMES, SEED, TEST_SIZE

SAVED_MODELS_DIR = Path("models")
EVAL_PATH = Path("results/processed/xgboost_eval.json")


def main():
    print("Loading training data...", flush=True)
    X, y, groups = build_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=groups,
    )
    print(f"Split: {len(X_train)} train, {len(X_test)} test", flush=True)

    print("Training XGBoost...", flush=True)
    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=SEED)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"XGBoost test MAE: {mae:.4f}", flush=True)

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SAVED_MODELS_DIR / "xgboost_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    eval_result = {
        "model": "xgboost",
        "mae": round(float(mae), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "y_test": [float(v) for v in y_test],
        "predictions": [float(v) for v in preds],
        "feature_importances": dict(zip(FEATURE_NAMES, [float(v) for v in model.feature_importances_])),
        "model_path": str(model_path),
    }
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2)

    print(f"Saved model -> {model_path}")
    print(f"Saved eval -> {EVAL_PATH}")


if __name__ == "__main__":
    main()