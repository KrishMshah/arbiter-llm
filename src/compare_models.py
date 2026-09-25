"""
src/compare_models.py — reads both eval files, picks the winner, writes
model_meta.json (what arbitrator.py loads), generates the plot.
Run after both training scripts: python -m src.compare_models
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

XGB_EVAL_PATH = Path("results/processed/xgboost_eval.json")
GBR_EVAL_PATH = Path("results/processed/gbr_eval.json")
MODEL_META_PATH = Path("models/model_meta.json")
PLOTS_DIR = Path("results/plots")


def plot_comparison(xgb_eval, gbr_eval):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(["XGBoost", "GradientBoosting"], [xgb_eval["mae"], gbr_eval["mae"]], color=["#1f77b4", "#ff7f0e"])
    ax1.set_ylabel("Test MAE (lower is better)")
    ax1.set_title("Model Comparison — Test MAE")
    for i, mae in enumerate([xgb_eval["mae"], gbr_eval["mae"]]):
        ax1.text(i, mae, f"{mae:.3f}", ha="center", va="bottom")

    lims = [1, 10]
    ax2.plot(lims, lims, "k--", alpha=0.4, label="Perfect prediction")
    ax2.scatter(xgb_eval["y_test"], xgb_eval["predictions"], alpha=0.5, label="XGBoost", color="#1f77b4", s=20)
    ax2.scatter(gbr_eval["y_test"], gbr_eval["predictions"], alpha=0.5, label="GradientBoosting", color="#ff7f0e", s=20)
    ax2.set_xlabel("Actual human_quality_score")
    ax2.set_ylabel("Predicted quality score")
    ax2.set_title("Predicted vs Actual (test set)")
    ax2.legend()
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "arbitrator_model_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved -> {out_path}")


def main():
    if not XGB_EVAL_PATH.exists() or not GBR_EVAL_PATH.exists():
        missing = [p for p in [XGB_EVAL_PATH, GBR_EVAL_PATH] if not p.exists()]
        print(f"Missing: {missing} -- run train_xgboost.py / train_gbr.py first.")
        return

    xgb_eval = json.loads(XGB_EVAL_PATH.read_text(encoding="utf-8"))
    gbr_eval = json.loads(GBR_EVAL_PATH.read_text(encoding="utf-8"))

    print(f"XGBoost           test MAE: {xgb_eval['mae']}")
    print(f"GradientBoosting  test MAE: {gbr_eval['mae']}")

    plot_comparison(xgb_eval, gbr_eval)

    winner = xgb_eval if xgb_eval["mae"] <= gbr_eval["mae"] else gbr_eval
    loser = gbr_eval if winner is xgb_eval else xgb_eval
    print(f"\nWinner: {winner['model']} (MAE {winner['mae']} vs {loser['mae']})")

    meta = {
        "winner": winner["model"],
        "winner_mae": winner["mae"],
        "xgboost_mae": xgb_eval["mae"],
        "gradient_boosting_mae": gbr_eval["mae"],
        "n_train": winner["n_train"],
        "n_test": winner["n_test"],
        "feature_importances": winner["feature_importances"],
        "model_path": winner["model_path"],
    }
    MODEL_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved winner meta -> {MODEL_META_PATH}")


if __name__ == "__main__":
    main()