"""
FinalScripts/selective_evaluation_curves.py

Selective evaluation in the sense of Trust or Escalate (Jung et al., ICLR 2025):
answer only when confident, abstain otherwise, and plot agreement with humans on the
answered items against coverage (share of items answered).

  Panel 1  Chatbot Arena, 85 non-tie human pairs: ARBITER's pointwise scores,
           confidence = absolute score gap (and, for comparison, the escalation
           confidence formula). Existing run, no new calls.
  Panel 2  TruthfulQA, 195 items with a ground-truth label: out-of-fold arbitrator
           prediction, confidence = distance of the predicted score from the
           midpoint 5.5 (and, for comparison, critic score variance).

Reference point: Trust or Escalate report about 80% agreement at about 80% coverage
on a Chatbot Arena subset with their own judges and pairwise prompts. That is a
different subset and judge protocol, so it is drawn as a marker, not a baseline.

Run: python FinalScripts/selective_evaluation_curves.py
"""

import numpy as np
import pandas as pd

from common import *
from feature_group_ablation import cv_predictions, xgb_cond

REF_COVERAGE, REF_AGREEMENT = 80.0, 80.0


def curve(conf, correct, min_n=15):
    """Keep the top-k most confident items for every k. conf: higher = more confident.
    Ties are kept together (a threshold either includes a tied group or not)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    n = len(conf)
    rows = []
    for tau in sorted(set(conf)):
        keep = conf >= tau
        k = int(keep.sum())
        if k < min_n:
            continue
        c = int(correct[keep].sum())
        lo, hi = wilson(c, k)
        rows.append({"min_confidence": tau, "n_answered": k, "coverage_pct": 100 * k / n,
                     "agreement_pct": 100 * c / k, "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi})
    return pd.DataFrame(rows).sort_values("coverage_pct").reset_index(drop=True)


def summarize(cv, name):
    full = cv.iloc[-1]
    near80 = cv.iloc[(cv["coverage_pct"] - REF_COVERAGE).abs().argmin()]
    hold = cv[cv["agreement_pct"] >= REF_AGREEMENT]
    best_cov = hold["coverage_pct"].max() if len(hold) else np.nan
    return {"curve": name,
            "agreement_at_full_coverage_pct": full["agreement_pct"],
            "n_total": int(full["n_answered"]),
            "agreement_near_80pct_coverage": near80["agreement_pct"],
            "coverage_there_pct": near80["coverage_pct"],
            "max_coverage_with_at_least_80pct_agreement": best_cov,
            "n_answered_at_that_point": int(hold.loc[hold["coverage_pct"].idxmax(), "n_answered"]) if len(hold) else np.nan}


def draw(ax, cv, color, label, band=True):
    ax.plot(cv["coverage_pct"], cv["agreement_pct"], "-o", color=color, ms=3.5, lw=1.4, label=label)
    if band:
        ax.fill_between(cv["coverage_pct"], cv["ci_lo_pct"], cv["ci_hi_pct"], color=color, alpha=0.12, lw=0)


def main():
    plt = setup_style()

    # ---------------- arena ----------------
    pp = pd.read_csv(ANALYSIS / "zheng_agreement_per_pair.csv")
    pp = pp[pp["outcome"] != "tie_vote_excluded_from_s2"].copy()
    pp["correct"] = pp["outcome"] == "correct"
    recs = load_jsonl(PROCESSED / "run_results_arena_reprocessed.jsonl")
    side = {}
    for r in recs:
        pid, s = r["input_id"].rsplit("_", 1)
        side.setdefault(pid, {})[s] = r["ml_arbitrator_output"]["arbitration_confidence"]
    pp["esc_conf"] = pp["pair_id"].map(lambda p: min(side[p].values()))
    n_ar = len(pp)
    print(f"[arena] {n_ar} non-tie pairs, full-coverage agreement {pp['correct'].mean()*100:.2f}%")

    arena_gap = curve(pp["abs_gap"], pp["correct"], min_n=10)
    arena_conf = curve(pp["esc_conf"], pp["correct"], min_n=10)

    # ---------------- truthfulqa ----------------
    df, items, by_id = build_dataset()
    preds = cv_predictions(df, xgb_cond(FEATURE_NAMES)).mean(axis=0)
    t = (df["dataset"] == "truthfulqa").values
    pred_untruthful = preds[t] < 5.5
    truth_untruthful = df.loc[t, "y"].values < 5
    ok = pred_untruthful == truth_untruthful
    tq_margin = curve(np.abs(preds[t] - 5.5), ok, min_n=20)
    tq_var = curve(-df.loc[t, "score_variance"].values, ok, min_n=20)
    print(f"[truthfulqa] {t.sum()} items, full-coverage agreement {ok.mean()*100:.2f}%")

    arena_gap.assign(curve="arena_abs_score_gap").pipe(
        lambda d: pd.concat([d, arena_conf.assign(curve="arena_escalation_confidence"),
                             tq_margin.assign(curve="truthfulqa_prediction_margin"),
                             tq_var.assign(curve="truthfulqa_critic_score_variance")])
    ).pipe(save_table, "selective_evaluation_agreement_vs_coverage_curves.csv")
    summ = pd.DataFrame([
        summarize(arena_gap, "Arena, confidence = absolute score gap"),
        summarize(arena_conf, "Arena, confidence = escalation formula"),
        summarize(tq_margin, "TruthfulQA, confidence = prediction margin"),
        summarize(tq_var, "TruthfulQA, confidence = critic score variance (low = confident)"),
    ])
    save_table(summ, "selective_evaluation_summary.csv")
    print(summ.round(2).to_string())

    # ---------------- plots ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    draw(ax, arena_gap, BLUE, "Confidence = absolute score gap")
    draw(ax, arena_conf, ORANGE, "Confidence = escalation formula", band=False)
    ax.axhline(50, color=LIGHTGREY, ls="--", lw=1)
    ax.text(3, 51, "chance (50%)", fontsize=8.5, color=GREY)
    ax.plot(REF_COVERAGE, REF_AGREEMENT, "D", color=RED, ms=8, zorder=5,
            label="Trust or Escalate (reported, ~80% / ~80%,\ntheir Arena subset and judges)")
    ax.set_title("Chatbot Arena: agreement with human votes vs coverage\n85 non-tie pairs, ARBITER pointwise scores", fontsize=10.5)
    ax.set_xlabel("Coverage: share of the 85 pairs the judge answers (%)")
    ax.set_ylabel("Agreement with human vote on answered pairs (%)")
    ax.set_xlim(0, 103)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", fontsize=8.5)

    ax = axes[1]
    draw(ax, tq_margin, BLUE, "Confidence = distance of predicted score from 5.5")
    draw(ax, tq_var, ORANGE, "Confidence = low critic score variance", band=False)
    ax.axhline(50, color=LIGHTGREY, ls="--", lw=1)
    ax.plot(REF_COVERAGE, REF_AGREEMENT, "D", color=RED, ms=8, zorder=5,
            label="Trust or Escalate reference (different task)")
    ax.set_title("TruthfulQA: agreement with the truthful/untruthful label vs coverage\n195 items, out-of-fold arbitrator predictions", fontsize=10.5)
    ax.set_xlabel("Coverage: share of the 195 items the judge answers (%)")
    ax.set_ylabel("Agreement with ground-truth label on answered items (%)")
    ax.set_xlim(0, 103)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Selective evaluation: answer only when confident (shaded bands = 95% Wilson intervals, first confidence signal)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "selective_evaluation_agreement_vs_coverage_arena_and_truthfulqa.png")
    plt.close(fig)


if __name__ == "__main__":
    main()