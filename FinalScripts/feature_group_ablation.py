"""
FinalScripts/feature_group_ablation.py

Question: do the disagreement features carry information about human quality
beyond what the critics' plain scores, severity and issue counts already tell us?

Method: repeated stratified 5-fold cross-validation (10 repeats, stratified by
dataset) on the 759 labelled items. Same XGBoost settings as src/train_xgboost.py.
Every number is out-of-fold. Intervals come from resampling items.

Run: python FinalScripts/feature_group_ablation.py
Writes: results/final/tables/ablation_*.csv, results/final/plots/ablation_*.png
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedStratifiedKFold
import xgboost as xgb

from common import *

warnings.filterwarnings("ignore")

N_SPLITS, N_REPEATS = 5, 10


def critic_mean_scores(by_id, ids):
    """Mean dimension score (1-5) per critic per item; NaN if the critic failed."""
    out = {c: [] for c in CRITIC_IDS}
    for i in ids:
        crit = {c["critic_id"]: c for c in by_id[i]["critiques"]}
        for cid in CRITIC_IDS:
            c = crit[cid]
            if c["critic_failed"] or not c["dimension_scores"]:
                out[cid].append(np.nan)
            else:
                out[cid].append(float(np.mean(list(c["dimension_scores"].values()))))
    return pd.DataFrame(out)


def make_xgb():
    return xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                            random_state=SEED, n_jobs=2, verbosity=0)


def cv_predictions(df, fit_predict, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    """Returns array (n_repeats, n_items) of out-of-fold predictions."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    strata = df["dataset"].values
    preds = np.full((n_repeats, len(df)), np.nan)
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(df)), strata)):
        r = k // n_splits
        preds[r, te] = fit_predict(df.iloc[tr], df.iloc[te])
    return preds


def xgb_cond(cols):
    def fp(tr, te):
        m = make_xgb()
        m.fit(tr[cols].values, tr["y"].values)
        return np.clip(m.predict(te[cols].values), 1, 10)
    return fp


def const_global(tr, te):
    return np.full(len(te), np.median(tr["y"]))


def per_group_median(group_col):
    def fp(tr, te):
        med = tr.groupby(group_col)["y"].median()
        return te[group_col].map(med).fillna(np.median(tr["y"])).values
    return fp


def linear_on(col):
    def fp(tr, te):
        ok = tr[col].notna()
        m = LinearRegression().fit(tr.loc[ok, [col]].values, tr.loc[ok, "y"].values)
        x = te[[col]].fillna(tr[col].mean()).values
        return np.clip(m.predict(x), 1, 10)
    return fp


def main():
    plt = setup_style()
    df, items, by_id = build_dataset()
    cm = critic_mean_scores(by_id, df["item_id"])
    for cid in CRITIC_IDS:
        df[f"mean_{cid}"] = cm[cid].values
    # the task_type column is categorical text; features hold the one-hot
    n = len(df)
    print(f"[ablation] {n} labelled items, {N_SPLITS}x{N_REPEATS} repeated CV")

    conditions = [
        # (label, kind, function)
        ("Predict overall median", "baseline", const_global),
        ("Predict median per task type", "baseline", per_group_median("task_type")),
        ("Predict median per dataset (uses dataset label)", "baseline", per_group_median("dataset")),
        ("Linear on mean score of all critics", "baseline", linear_on("score_mean")),
        (f"Linear on {CRITIC_LABEL['critic_a']} alone", "baseline", linear_on("mean_critic_a")),
        (f"Linear on {CRITIC_LABEL['critic_b']} alone", "baseline", linear_on("mean_critic_b")),
        (f"Linear on {CRITIC_LABEL['critic_c']} alone", "baseline", linear_on("mean_critic_c")),
        ("XGBoost: task type only", "model", xgb_cond(TASK_FEATURES)),
        ("XGBoost: disagreement features only (9)", "model", xgb_cond(DISAGREEMENT_FEATURES)),
        ("XGBoost: critic-signal features only (6)", "model", xgb_cond(CRITIC_SIGNAL_FEATURES)),
        ("XGBoost: critic-signal + task type (10)", "model", xgb_cond(CRITIC_SIGNAL_FEATURES + TASK_FEATURES)),
        ("XGBoost: disagreement + task type (13)", "model", xgb_cond(DISAGREEMENT_FEATURES + TASK_FEATURES)),
        ("XGBoost: all 19 features", "model", xgb_cond(FEATURE_NAMES)),
    ]

    y = df["y"].values
    results, per_item_err, mean_pred = [], {}, {}
    for label, kind, fp in conditions:
        preds = cv_predictions(df, fp)
        err = np.abs(preds - y[None, :]).mean(axis=0)          # per-item, averaged over repeats
        per_item_err[label] = err
        mean_pred[label] = preds.mean(axis=0)
        lo, hi = bootstrap_ci(err)
        results.append({"condition": label, "kind": kind, "mae": err.mean(), "ci_lo": lo, "ci_hi": hi})
        print(f"  {label:58s} MAE {err.mean():.3f}  [{lo:.3f}, {hi:.3f}]")

    res = pd.DataFrame(results)

    # ---- paired differences that matter for the claim --------------------
    all19 = per_item_err["XGBoost: all 19 features"]
    pairs = [
        ("All 19 vs critic-signal + task (adds disagreement features)", "XGBoost: critic-signal + task type (10)"),
        ("All 19 vs disagreement + task (adds critic-signal features)", "XGBoost: disagreement + task type (13)"),
        ("All 19 vs median per dataset", "Predict median per dataset (uses dataset label)"),
        ("All 19 vs linear on mean critic score", "Linear on mean score of all critics"),
        ("All 19 vs task type only", "XGBoost: task type only"),
        ("Disagreement + task vs task type only", None),
    ]
    diffs = []
    for name, other in pairs:
        if other is None:
            a, b = per_item_err["XGBoost: disagreement + task type (13)"], per_item_err["XGBoost: task type only"]
        else:
            a, b = all19, per_item_err[other]
        d, lo, hi = paired_bootstrap_diff(a, b)
        diffs.append({"comparison": name, "mae_difference": d, "ci_lo": lo, "ci_hi": hi,
                      "reading": "negative = first is better"})
        print(f"  Δ {name}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    diffs = pd.DataFrame(diffs)

    # ---- per-dataset MAE and within-dataset rank correlation -----------------
    key = ["Predict median per dataset (uses dataset label)",
           "Linear on mean score of all critics",
           "XGBoost: critic-signal + task type (10)",
           "XGBoost: disagreement + task type (13)",
           "XGBoost: all 19 features"]
    rows = []
    for ds in DATASET_LABEL:
        mask = (df["dataset"] == ds).values
        for label in key:
            e = per_item_err[label][mask]
            lo, hi = bootstrap_ci(e)
            rho = spearmanr(mean_pred[label][mask], y[mask]).correlation
            rows.append({"dataset": ds, "condition": label, "n": int(mask.sum()),
                         "mae": e.mean(), "ci_lo": lo, "ci_hi": hi,
                         "spearman_within_dataset": rho})
    per_ds = pd.DataFrame(rows)

    save_table(res, "ablation_feature_groups_heldout_mae.csv")
    save_table(diffs, "ablation_feature_groups_paired_differences.csv")
    save_table(per_ds, "ablation_feature_groups_by_dataset.csv")

    # ---- plot 1: overall MAE with intervals -----------------------------------
    order = res.sort_values("mae", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ypos = np.arange(len(order))
    for i, r in order.iterrows():
        color = GREY if r["kind"] == "baseline" else BLUE
        ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=color, lw=1.4)
        ax.plot(r["mae"], i, "o", color=color, ms=6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order["condition"])
    ax.set_xlabel("Mean absolute error on the 1-10 human quality score (lower is better)")
    ax.set_title("Feature-group ablation: held-out error of the quality-score model\n"
                 f"{N_SPLITS}-fold cross-validation repeated {N_REPEATS} times, {n} labelled items, 95% intervals over items",
                 fontsize=10.5)
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", color=GREY, label="simple baseline")
    ax.plot([], [], "o", color=BLUE, label="XGBoost model")
    ax.legend(loc="lower left")
    save_fig(fig, "ablation_feature_groups_heldout_error_all_data.png")
    plt.close(fig)

    # ---- plot 2: per dataset ----------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), sharex=False)
    short = {
        "Predict median per dataset (uses dataset label)": "Median per dataset",
        "Linear on mean score of all critics": "Mean critic score",
        "XGBoost: critic-signal + task type (10)": "Critic-signal + task",
        "XGBoost: disagreement + task type (13)": "Disagreement + task",
        "XGBoost: all 19 features": "All 19 features",
    }
    for ax, ds in zip(axes, DATASET_LABEL):
        sub = per_ds[per_ds["dataset"] == ds].set_index("condition").loc[key]
        yp = np.arange(len(key))[::-1]
        for yy, (cond, r) in zip(yp, sub.iterrows()):
            color = GREY if cond in key[:2] else BLUE
            ax.plot([r["ci_lo"], r["ci_hi"]], [yy, yy], color=color, lw=1.4)
            ax.plot(r["mae"], yy, "o", color=color, ms=5.5)
        ax.set_yticks(yp)
        ax.set_yticklabels([short[k] for k in key] if ds == "mt_bench_human" else [])
        ax.set_title(f"{DATASET_LABEL[ds]} (n={int(sub['n'].iloc[0])})", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("MAE (1-10 scale)")
    fig.suptitle("Feature-group ablation by dataset: held-out error of the quality-score model (grey = simple baselines, blue = XGBoost)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, "ablation_feature_groups_heldout_error_by_dataset.png")
    plt.close(fig)


if __name__ == "__main__":
    main()