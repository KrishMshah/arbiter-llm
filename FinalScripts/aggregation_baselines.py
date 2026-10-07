"""
FinalScripts/aggregation_baselines.py

Question: does the learned arbitrator beat the simple ways of pooling a panel of
critics (PoLL-style averaging, median, minimum, majority vote)?

Part A  quality score, all 759 items, repeated stratified CV (same folds as the
        ablation). Each pooling rule is turned into a score by a linear map fitted
        on the training folds only, so every number is out-of-fold.
Part B  detection on the two datasets with ground-truth labels:
        TruthfulQA (untruthful vs truthful, n=195, balanced) and FActScore
        (lower-factuality half vs upper half by supported-claim ratio, n=174).
        Votes are scored by accuracy; continuous scores by AUC.

Run: python FinalScripts/aggregation_baselines.py
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import *
from feature_group_ablation import (critic_mean_scores, cv_predictions, xgb_cond,
                                       linear_on, N_SPLITS, N_REPEATS)

warnings.filterwarnings("ignore")

SERIOUS = {"major", "critical"}


def critic_flags(by_id, ids):
    """1 if the critic reports at least one major/critical issue (and did not fail)."""
    out = {c: [] for c in CRITIC_IDS}
    for i in ids:
        crit = {c["critic_id"]: c for c in by_id[i]["critiques"]}
        for cid in CRITIC_IDS:
            c = crit[cid]
            if c["critic_failed"]:
                out[cid].append(np.nan)
            else:
                out[cid].append(float(any(x.get("severity") in SERIOUS for x in c["issues"])))
    return pd.DataFrame(out)


def auc_ci(score, label, n_boot=2000, seed=SEED):
    score, label = np.asarray(score), np.asarray(label)
    a = roc_auc_score(label, score)
    rng = np.random.default_rng(seed)
    n = len(score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if label[idx].min() == label[idx].max():
            continue
        vals.append(roc_auc_score(label[idx], score[idx]))
    return a, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def spearman_ci(x, y, n_boot=2000, seed=SEED):
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(x)
    r = spearmanr(x, y).correlation
    vals = [spearmanr(x[i], y[i]).correlation for i in rng.integers(0, n, (n_boot, n))]
    return float(r), float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def mcnemar_exact(correct_a, correct_b):
    from scipy.stats import binomtest
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = 1.0 if n01 + n10 == 0 else binomtest(n10, n10 + n01, 0.5).pvalue
    return n10, n01, p


def main():
    plt = setup_style()
    df, items, by_id = build_dataset()
    ids = df["item_id"]
    cm = critic_mean_scores(by_id, ids)
    fl = critic_flags(by_id, ids)
    for cid in CRITIC_IDS:
        df[f"mean_{cid}"] = cm[cid].values
        df[f"flag_{cid}"] = fl[cid].values
    df["pool_mean"] = cm.mean(axis=1).values
    df["pool_median"] = cm.median(axis=1).values
    df["pool_min"] = cm.min(axis=1).values
    df["pool_max"] = cm.max(axis=1).values
    df["vote_count"] = fl.sum(axis=1).values            # how many critics flag a serious issue
    y = df["y"].values
    n = len(df)

    # ================= Part A: quality-score regression =================
    rules = [
        ("Average of critic scores (PoLL-style mean)", "pool_mean"),
        ("Median of critic scores", "pool_median"),
        ("Lowest critic score", "pool_min"),
        ("Highest critic score", "pool_max"),
        ("Number of critics flagging a serious issue (vote count)", "vote_count"),
        (f"{CRITIC_LABEL['critic_b']} alone", "mean_critic_b"),
    ]
    from sklearn.linear_model import LinearRegression

    def linear_multi(cols):
        def fp(tr, te):
            m = LinearRegression().fit(tr[cols].values, tr["y"].values)
            return np.clip(m.predict(te[cols].fillna(tr[cols].mean()).values), 1, 10)
        return fp

    fair = [
        ("Average + task type (linear)", linear_multi(["pool_mean"] + TASK_FEATURES)),
        ("Average + task type (XGBoost)", xgb_cond(["pool_mean"] + TASK_FEATURES)),
    ]
    err, meanpred, rows = {}, {}, []
    for label, col in rules + fair + [("Learned arbitrator (XGBoost, 19 features)", None)]:
        if col is None:
            fp = xgb_cond(FEATURE_NAMES)
        elif callable(col):
            fp = col
        else:
            fp = linear_on(col)
        preds = cv_predictions(df, fp)
        e = np.abs(preds - y[None, :]).mean(axis=0)
        err[label], meanpred[label] = e, preds.mean(axis=0)
        lo, hi = bootstrap_ci(e)
        rows.append({"method": label, "mae": e.mean(), "ci_lo": lo, "ci_hi": hi})
        print(f"  {label:62s} MAE {e.mean():.3f} [{lo:.3f}, {hi:.3f}]")
    res = pd.DataFrame(rows)

    learned = "Learned arbitrator (XGBoost, 19 features)"
    drows = []
    for label, _ in rules + fair:
        d, lo, hi = paired_bootstrap_diff(err[learned], err[label])
        drows.append({"comparison": f"Learned arbitrator vs {label}", "mae_difference": d,
                      "ci_lo": lo, "ci_hi": hi, "reading": "negative = learned arbitrator better"})
    drows = pd.DataFrame(drows)
    save_table(res, "aggregation_pooling_vs_learned_heldout_mae.csv")
    save_table(drows, "aggregation_pooling_vs_learned_paired_differences.csv")

    # within-dataset rank correlation (does it order items inside a dataset?)
    srows = []
    for ds in DATASET_LABEL:
        m = (df["dataset"] == ds).values
        for label in [r[0] for r in rules] + [learned]:
            r, lo, hi = spearman_ci(meanpred[label][m], y[m])
            srows.append({"dataset": ds, "method": label, "n": int(m.sum()),
                          "spearman": r, "ci_lo": lo, "ci_hi": hi})
    srows = pd.DataFrame(srows)
    save_table(srows, "aggregation_pooling_vs_learned_spearman_by_dataset.csv")

    # ================= Part B: detection with ground truth =================
    det_rows, vote_rows = [], []
    arb = meanpred[learned]
    specs = {}
    t = (df["dataset"] == "truthfulqa").values
    specs["truthfulqa"] = (t, (df.loc[t, "y"] < 5).astype(int).values,
                           "untruthful answer (label from TruthfulQA)")
    f = (df["dataset"] == "factscore_labeled").values
    thr = np.median(df.loc[f, "y"])
    specs["factscore_labeled"] = (f, (df.loc[f, "y"] < thr).astype(int).values,
                                  f"lower-factuality half (score < {thr:.2f})")
    for ds, (m, lab, desc) in specs.items():
        sub = df[m]
        print(f"[detect] {ds}: n={m.sum()}, positives={lab.sum()} -> {desc}")
        scores = {
            "Average of critic scores": -sub["pool_mean"].values,
            "Median of critic scores": -sub["pool_median"].values,
            "Lowest critic score": -sub["pool_min"].values,
            f"{CRITIC_LABEL['critic_a']} score": -sub["mean_critic_a"].values,
            f"{CRITIC_LABEL['critic_b']} score": -sub["mean_critic_b"].values,
            f"{CRITIC_LABEL['critic_c']} score": -sub["mean_critic_c"].values,
            "Learned arbitrator (out-of-fold)": -arb[m],
        }
        for name, s in scores.items():
            s = np.where(np.isnan(s), np.nanmean(s), s)
            a, lo, hi = auc_ci(s, lab)
            det_rows.append({"dataset": ds, "n": int(m.sum()), "method": name,
                             "auc": a, "ci_lo": lo, "ci_hi": hi})
            print(f"    AUC {name:40s} {a:.3f} [{lo:.3f}, {hi:.3f}]")

        # votes (flag = at least one major/critical issue)
        fcols = {CRITIC_LABEL[c]: sub[f"flag_{c}"].fillna(0).values for c in CRITIC_IDS}
        votes = sub["vote_count"].fillna(0).values
        preds = {
            **{f"{k} flags": v for k, v in fcols.items()},
            "Majority vote (2 of 3)": (votes >= 2).astype(float),
            "Any critic flags": (votes >= 1).astype(float),
            "All critics flag": (votes >= 3).astype(float),
        }
        correct = {}
        for name, p in preds.items():
            ok = (p == lab)
            correct[name] = ok
            acc = ok.mean()
            lo, hi = wilson(int(ok.sum()), len(ok))
            tpr = p[lab == 1].mean()
            fpr = p[lab == 0].mean()
            vote_rows.append({"dataset": ds, "n": int(m.sum()), "rule": name, "accuracy": acc,
                              "ci_lo": lo, "ci_hi": hi, "flag_rate_on_positives": tpr,
                              "flag_rate_on_negatives": fpr})
            print(f"    ACC {name:36s} {acc:.3f} [{lo:.3f}, {hi:.3f}]  flags pos {tpr:.2f} neg {fpr:.2f}")
        best = max([k for k in preds if k.endswith("flags")], key=lambda k: correct[k].mean())
        n10, n01, p = mcnemar_exact(correct["Majority vote (2 of 3)"], correct[best])
        print(f"    McNemar majority vs {best}: majority-only right {n10}, best-only right {n01}, p={p:.3f}")
        vote_rows.append({"dataset": ds, "n": int(m.sum()),
                          "rule": f"McNemar: majority vote vs {best}",
                          "accuracy": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                          "flag_rate_on_positives": n10, "flag_rate_on_negatives": n01,
                          "mcnemar_p": p})
    det = pd.DataFrame(det_rows)
    votes_df = pd.DataFrame(vote_rows)
    save_table(det, "aggregation_detection_auc_truthfulqa_factscore.csv")
    save_table(votes_df, "aggregation_detection_votes_truthfulqa_factscore.csv")

    # ================= plots =================
    order = res.sort_values("mae", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    for i, r in order.iterrows():
        color = BLUE if r["method"] == learned else GREY
        ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=color, lw=1.4)
        ax.plot(r["mae"], i, "o", color=color, ms=6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order["method"])
    ax.set_xlabel("Mean absolute error on the 1-10 human quality score (lower is better)")
    ax.set_title("Pooling the critics vs the learned arbitrator: held-out error of the quality score\n"
                 f"{N_SPLITS}-fold CV repeated {N_REPEATS} times, {n} labelled items, 95% intervals over items",
                 fontsize=10.5)
    ax.grid(axis="y", visible=False)
    save_fig(fig, "aggregation_pooling_vs_learned_arbitrator_heldout_error.png")
    plt.close(fig)

    # per-dataset Spearman panels
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.7), sharey=False)
    labels = [r[0] for r in rules[:5]] + [learned]
    short = {rules[0][0]: "Average", rules[1][0]: "Median", rules[2][0]: "Lowest score",
             rules[3][0]: "Highest score", rules[4][0]: "Vote count", learned: "Learned arbitrator"}
    for ax, ds in zip(axes, DATASET_LABEL):
        sub = srows[srows["dataset"] == ds].set_index("method").loc[labels]
        yp = np.arange(len(labels))[::-1]
        for yy, (name, r) in zip(yp, sub.iterrows()):
            c = BLUE if name == learned else GREY
            ax.plot([r["ci_lo"], r["ci_hi"]], [yy, yy], color=c, lw=1.4)
            ax.plot(r["spearman"], yy, "o", color=c, ms=5.5)
        ax.axvline(0, color="#999999", lw=0.8)
        ax.set_yticks(yp)
        ax.set_yticklabels([short[k] for k in labels] if ds == "mt_bench_human" else [])
        ax.set_title(f"{DATASET_LABEL[ds]} (n={int(sub['n'].iloc[0])})", fontsize=10)
        ax.set_xlabel("Spearman correlation with human score")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Ordering items within each dataset: pooled critic scores vs learned arbitrator (held-out predictions, 95% intervals)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, "aggregation_pooling_vs_learned_arbitrator_rank_correlation_by_dataset.png")
    plt.close(fig)

    # detection AUC plot
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=False)
    names = list(det[det["dataset"] == "truthfulqa"]["method"])
    for ax, ds, ttl in zip(axes, ["truthfulqa", "factscore_labeled"],
                           ["TruthfulQA: flag the untruthful answer (n=195)",
                            "FActScore: flag the lower-factuality half (n=174)"]):
        sub = det[det["dataset"] == ds].set_index("method").loc[names]
        yp = np.arange(len(names))[::-1]
        for yy, (name, r) in zip(yp, sub.iterrows()):
            c = BLUE if name.startswith("Learned") else GREY
            ax.plot([r["ci_lo"], r["ci_hi"]], [yy, yy], color=c, lw=1.4)
            ax.plot(r["auc"], yy, "o", color=c, ms=5.5)
        ax.axvline(0.5, color="#999999", lw=0.8, ls="--")
        ax.set_yticks(yp)
        ax.set_yticklabels(names if ds == "truthfulqa" else [])
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("AUC (0.5 = chance)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Detecting bad answers with ground-truth labels: AUC of each critic, pooled scores and the learned arbitrator (95% intervals)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, "aggregation_detection_auc_truthfulqa_and_factscore.png")
    plt.close(fig)


if __name__ == "__main__":
    main()