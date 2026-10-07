"""
FinalScripts/critic_subset_panels.py

Question: is each critic worth paying for?  Haiku is ~97% of the critic spend.

Method: for every non-empty subset of the three critics, rebuild the 19 features
from the stored critiques of just those critics (src.disagreement, the same code
the pipeline uses, including ground-truth gating), train the same XGBoost with the
same repeated stratified CV, and compare held-out error. Cost per item is the
critics' measured API cost (arena run, API-reported tokens); the local Llama is free.

Run: python FinalScripts/critic_subset_panels.py
"""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import *
from src.schemas import CritiqueOutput, BenchmarkItem
from src.disagreement import detect_disagreement, extract_ml_features
from feature_group_ablation import cv_predictions, xgb_cond, N_SPLITS, N_REPEATS
from aggregation_baselines import auc_ci

warnings.filterwarnings("ignore")

SHORT = {"critic_a": "GPT-4o-mini", "critic_b": "Haiku 4.5", "critic_c": "Llama 3.2 (local)"}


def subset_label(sub):
    return " + ".join(SHORT[c] for c in sub)


def features_for(subset, rec, item):
    crits = [CritiqueOutput(**c) for c in rec["critiques"] if c["critic_id"] in subset]
    bench = BenchmarkItem(**item)
    matrix = detect_disagreement(crits, benchmark_item=bench)
    f = extract_ml_features(crits, matrix, rec["task_type"])
    d = f.model_dump() if hasattr(f, "model_dump") else dict(f)
    return {k: float(d[k]) for k in FEATURE_NAMES}


def main():
    plt = setup_style()
    df0, items, by_id = build_dataset()
    ids = df0["item_id"].tolist()
    costs = measured_critic_costs()
    print("[cost per item, measured]", {k: round(v, 5) for k, v in costs.items()})

    # sanity: rebuilding with all three critics must reproduce the stored features
    bad = 0
    for iid in ids[:60]:
        new = features_for(set(CRITIC_IDS), by_id[iid], items[iid])
        old = by_id[iid]["ml_features"]
        if any(abs(new[k] - float(old[k])) > 1e-6 for k in FEATURE_NAMES):
            bad += 1
    print(f"[sanity] items whose rebuilt features differ from stored (of 60): {bad}")

    y = df0["y"].values
    subsets = [s for r in (3, 2, 1) for s in combinations(CRITIC_IDS, r)]
    rows, drows, srows, errs, preds_mean = [], [], [], {}, {}
    for sub in subsets:
        feats = [features_for(set(sub), by_id[i], items[i]) for i in ids]
        d = pd.concat([df0[["item_id", "dataset", "task_type", "y"]].reset_index(drop=True),
                       pd.DataFrame(feats)], axis=1)
        preds = cv_predictions(d, xgb_cond(FEATURE_NAMES))
        e = np.abs(preds - y[None, :]).mean(axis=0)
        lab = subset_label(sub)
        errs[lab], preds_mean[lab] = e, preds.mean(axis=0)
        lo, hi = bootstrap_ci(e)
        cost = sum(costs[c] for c in sub)
        rows.append({"panel": lab, "n_critics": len(sub), "dollars_per_item": cost,
                     "mae": e.mean(), "ci_lo": lo, "ci_hi": hi})
        print(f"  {lab:46s} ${cost:.5f}/item  MAE {e.mean():.3f} [{lo:.3f}, {hi:.3f}]")
        for ds in DATASET_LABEL:
            m = (d["dataset"] == ds).values
            srows.append({"panel": lab, "dataset": ds, "n": int(m.sum()),
                          "mae": e[m].mean(),
                          "spearman": spearmanr(preds_mean[lab][m], y[m]).correlation})
    res = pd.DataFrame(rows)

    full = subset_label(CRITIC_IDS)
    for lab in errs:
        if lab == full:
            continue
        dd, lo, hi = paired_bootstrap_diff(errs[lab], errs[full])
        drows.append({"panel": lab, "mae_minus_full_panel": dd, "ci_lo": lo, "ci_hi": hi,
                      "reading": "positive = worse than the full three-critic panel"})
    drows = pd.DataFrame(drows)

    # TruthfulQA detection AUC per panel
    t = (df0["dataset"] == "truthfulqa").values
    lab_t = (y[t] < 5).astype(int)
    arows = []
    for lab in errs:
        a, lo, hi = auc_ci(-preds_mean[lab][t], lab_t)
        arows.append({"panel": lab, "auc_truthfulqa": a, "ci_lo": lo, "ci_hi": hi})
        print(f"  AUC TruthfulQA {lab:46s} {a:.3f} [{lo:.3f}, {hi:.3f}]")
    arows = pd.DataFrame(arows)
    res = res.merge(arows[["panel", "auc_truthfulqa", "ci_lo", "ci_hi"]].rename(
        columns={"ci_lo": "auc_ci_lo", "ci_hi": "auc_ci_hi"}), on="panel")

    save_table(res, "critic_subsets_heldout_error_and_cost.csv")
    save_table(drows, "critic_subsets_paired_difference_vs_full_panel.csv")
    save_table(pd.DataFrame(srows), "critic_subsets_by_dataset.csv")

    # ---- plot: one row per panel, cost written in the label ----
    order = res.sort_values(["dollars_per_item", "mae"], ascending=[False, False]).reset_index(drop=True)
    names = [f"{r['panel']}  (${r['dollars_per_item']:.4f}/item)" for _, r in order.iterrows()]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for ax, (ycol, lo, hi, xlab, ttl) in zip(axes, [
        ("mae", "ci_lo", "ci_hi", "Mean absolute error on the 1-10 human score (lower is better)",
         "Quality-score error, all 759 items"),
        ("auc_truthfulqa", "auc_ci_lo", "auc_ci_hi", "AUC for spotting untruthful answers (higher is better)",
         "TruthfulQA detection, 195 items")]):
        for i, (_, r) in enumerate(order.iterrows()):
            c = BLUE if r["n_critics"] == 3 else GREY
            ax.plot([r[lo], r[hi]], [i, i], color=c, lw=1.4)
            ax.plot(r[ycol], i, "o", color=c, ms=6)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(names if ax is axes[0] else [])
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_title(ttl, fontsize=10.5)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Is each critic worth paying for? Critic subsets rebuilt from stored critiques, with cost per item "
                 "(blue = all three critics, 95% intervals)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, "critic_subsets_error_vs_dollars_per_item.png")
    plt.close(fig)


if __name__ == "__main__":
    main()