"""
FinalScripts/rq1_to_rq5_research_questions.py      

Computes the five research questions of the project from results that already exist.
All arbitrator numbers use the 17 label-free features (the script refuses to run otherwise).

  RQ1  Do disagreement rates differ across critic pairs (GPT-4o-mini, Haiku 4.5, Llama 3.2 3B)?
  RQ2  Does disagreement frequency correlate with the human quality score?
  RQ3  Are some quality dimensions more disagreement-prone, and does that vary by task type?
  RQ4  Can the learned arbitrator match a strong-model (GPT-5.6) judgement at much lower cost?
  RQ5  Does the tracer find the right origin sentence, and how far do dependents propagate?

Definitions used throughout
  * Two critics "disagree" on an item when their scores differ by 2 or more on at least one shared
    dimension (the pipeline's own SCORE_GAP_THRESHOLD). Critics that failed on an item are skipped.
  * The same 759 labelled items as every other analysis (MT-Bench 195, SummEval 195, TruthfulQA 195,
    FActScore 174).
  * The three critics score the SAME items, so tests for RQ1 and for the dimension comparison in RQ3
    are paired (Cochran's Q / exact McNemar / Friedman / Wilcoxon, Holm-corrected). A one-way ANOVA
    over pairs is also printed for RQ1 only because the project plan named it; it ignores the pairing.
  * Intervals: Wilson for proportions, bootstrap over items (2000 resamples) for everything else.

RQ4 needs results/processed/pointwise_terra_759.jsonl (the GPT-5.6 run). If it is missing, RQ4 is
computed without the GPT-5.6 comparison. GPT-5.6 results are used ONLY for MT-Bench and SummEval,
because that run was shown an arbitrator estimate built with the two label-derived features, which
inflated the arbitrator on TruthfulQA and FActScore.

RQ5 reads the tracer stored in run_results_combined_costed.jsonl and the validation tables written
earlier by tracer_origin_validation.py (results/final/tables/tracer_*.csv).

Run from the repo root:  python FinalScripts/rq1_to_rq5_research_questions.py
Writes: results/rq/*.csv and results/rq/*.png
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import (binomtest, chi2, chi2_contingency, f_oneway, friedmanchisquare,
                         kruskal, rankdata, wilcoxon)

from common import *
from feature_group_ablation import cv_predictions, linear_on, xgb_cond

warnings.filterwarnings("ignore")

if not LABEL_FREE:
    raise SystemExit("Run without ARBITER_LABEL_FREE=0: the RQ analysis must use the label-free features.")

RQ_DIR = ROOT / "results" / "rq"
RQ_DIR.mkdir(parents=True, exist_ok=True)
GAP = 2                       # score gap that counts as a disagreement
N_BOOT = 2000
SHORT = {"critic_a": "GPT-4o-mini", "critic_b": "Haiku 4.5", "critic_c": "Llama 3.2 3B"}
PAIRS = [("critic_a", "critic_b"), ("critic_a", "critic_c"), ("critic_b", "critic_c")]
PAIR_LABEL = {p: f"{SHORT[p[0]]} vs {SHORT[p[1]]}" for p in PAIRS}
TASK_LABEL = {"factual_qa": "Factual QA", "summarisation": "Summarisation",
              "reasoning": "Reasoning", "creative": "Creative"}
DIM_LABEL = {"factual_accuracy": "Factual accuracy", "logical_consistency": "Logical consistency",
             "completeness": "Completeness"}
HEADLINE = []                 # rows for rq_headline_numbers.csv


def save_t(df, name):
    p = RQ_DIR / name
    df.to_csv(p, index=False)
    print(f"[table] {p.relative_to(ROOT)}")


def save_f(fig, name):
    p = RQ_DIR / name
    fig.savefig(p)
    print(f"[plot]  {p.relative_to(ROOT)}")


def headline(rq, metric, value, lo=None, hi=None, note=""):
    HEADLINE.append({"rq": rq, "metric": metric, "value": value, "ci_lo": lo, "ci_hi": hi, "note": note})


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj, run = np.empty(m), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def cochran_q(X):
    """X: n x k binary matrix of paired outcomes. Returns (Q, p)."""
    X = np.asarray(X, float)
    k = X.shape[1]
    cj, ri, n_tot = X.sum(0), X.sum(1), X.sum()
    denom = k * n_tot - (ri ** 2).sum()
    if denom == 0:
        return np.nan, np.nan
    q = (k - 1) * (k * (cj ** 2).sum() - n_tot ** 2) / denom
    return float(q), float(chi2.sf(q, k - 1))


def mcnemar_exact(x, y):
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    return (b, c, 1.0) if b + c == 0 else (b, c, float(binomtest(b, b + c, 0.5).pvalue))


def safe_wilcoxon(a, b):
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:                       # all differences zero
        return 1.0


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ============================================================================== data
def critic_dim_scores(rec):
    """critic_id -> {dimension: score} for critics that succeeded."""
    out = {}
    for c in rec["critiques"]:
        if c["critic_failed"] or not c["dimension_scores"]:
            continue
        out[c["critic_id"]] = {k: float(v) for k, v in c["dimension_scores"].items()}
    return out


def build_tables():
    df, items, by_id = build_dataset()
    pair_rows, dim_rows = [], []
    for _, r in df.iterrows():
        sc = critic_dim_scores(by_id[r["item_id"]])
        base = {"item_id": r["item_id"], "dataset": r["dataset"], "task_type": r["task_type"]}
        pr = dict(base)
        for p in PAIRS:
            if p[0] in sc and p[1] in sc:
                shared = sorted(set(sc[p[0]]) & set(sc[p[1]]))
                if shared:
                    gaps = np.array([abs(sc[p[0]][d] - sc[p[1]][d]) for d in shared])
                    pr[f"D_{p[0]}_{p[1]}"] = float((gaps >= GAP).any())
                    pr[f"G_{p[0]}_{p[1]}"] = float(gaps.mean())
        pair_rows.append(pr)
        dims = sorted({d for c in sc.values() for d in c})
        for d in dims:
            vals = [sc[c][d] for c in sc if d in sc[c]]
            if len(vals) >= 2:
                dim_rows.append({**base, "dimension": d, "n_critics": len(vals),
                                 "score_range": max(vals) - min(vals), "D": float(max(vals) - min(vals) >= GAP)})
    return df, pd.DataFrame(pair_rows), pd.DataFrame(dim_rows), by_id


# ============================================================================== RQ1
def rq1(pairs_df):
    print("\n=== RQ1: disagreement rate by critic pair ===")
    cols_d = [f"D_{a}_{b}" for a, b in PAIRS]
    cols_g = [f"G_{a}_{b}" for a, b in PAIRS]
    comp = pairs_df.dropna(subset=cols_d).copy()
    print(f"  items with all three critics present: {len(comp)} of {len(pairs_df)}")
    scopes = [("All items", comp)] + [(DATASET_LABEL[d], comp[comp["dataset"] == d]) for d in DATASET_LABEL]
    rate_rows, test_rows = [], []
    for name, sub in scopes:
        n = len(sub)
        if n < 5:
            continue
        for p, cd, cg in zip(PAIRS, cols_d, cols_g):
            k = int(sub[cd].sum())
            lo, hi = wilson(k, n)
            glo, ghi = bootstrap_ci(sub[cg].values, n_boot=N_BOOT)
            rate_rows.append({"scope": name, "pair": PAIR_LABEL[p], "n_items": n, "items_disagreeing": k,
                              "rate": k / n, "rate_ci_lo": lo, "rate_ci_hi": hi,
                              "mean_abs_gap": sub[cg].mean(), "gap_ci_lo": glo, "gap_ci_hi": ghi})
        X = sub[cols_d].values
        q, p = cochran_q(X)
        test_rows.append({"scope": name, "test": "Cochran Q on 'any gap >= 2' (3 pairs, paired)", "n_items": n,
                          "statistic": q, "p_value": p, "p_holm": np.nan, "detail": ""})
        try:
            fr = friedmanchisquare(*[sub[c].values for c in cols_g])
            test_rows.append({"scope": name, "test": "Friedman on mean absolute gap (3 pairs, paired)", "n_items": n,
                              "statistic": float(fr.statistic), "p_value": float(fr.pvalue), "p_holm": np.nan,
                              "detail": ""})
        except ValueError:
            pass
        an = f_oneway(*[sub[c].values for c in cols_g])
        test_rows.append({"scope": name, "test": "One-way ANOVA on mean absolute gap (reference only, ignores pairing)",
                          "n_items": n, "statistic": float(an.statistic), "p_value": float(an.pvalue),
                          "p_holm": np.nan, "detail": ""})
        pw = []
        for (i, ci), (j, cj) in itertools.combinations(list(enumerate(cols_d)), 2):
            b, c, pm = mcnemar_exact(sub[ci].values, sub[cj].values)
            pw.append((PAIRS[i], PAIRS[j], b, c, pm, safe_wilcoxon(sub[cols_g[i]].values, sub[cols_g[j]].values)))
        hm = holm([x[4] for x in pw])
        hw = holm([x[5] for x in pw])
        for (pa, pb, b, c, pm, pwx), a1, a2 in zip(pw, hm, hw):
            test_rows.append({"scope": name, "test": f"Exact McNemar: {PAIR_LABEL[pa]} vs {PAIR_LABEL[pb]}",
                              "n_items": n, "statistic": np.nan, "p_value": pm, "p_holm": a1,
                              "detail": f"only first disagrees {b}, only second disagrees {c}"})
            test_rows.append({"scope": name, "test": f"Wilcoxon on mean gap: {PAIR_LABEL[pa]} vs {PAIR_LABEL[pb]}",
                              "n_items": n, "statistic": np.nan, "p_value": pwx, "p_holm": a2, "detail": ""})
    rates, tests = pd.DataFrame(rate_rows), pd.DataFrame(test_rows)
    save_t(rates, "rq1_disagreement_rate_by_critic_pair.csv")
    save_t(tests, "rq1_tests_across_critic_pairs.csv")

    allr = rates[rates["scope"] == "All items"]
    for _, r in allr.iterrows():
        print(f"  {r['pair']:30s} disagree {100*r['rate']:5.1f}% [{100*r['rate_ci_lo']:.1f}, {100*r['rate_ci_hi']:.1f}]"
              f"  mean gap {r['mean_abs_gap']:.2f}")
        headline("RQ1", f"disagreement rate, {r['pair']}", r["rate"], r["rate_ci_lo"], r["rate_ci_hi"], "all items")
    qrow = tests[(tests["scope"] == "All items") & tests["test"].str.startswith("Cochran")].iloc[0]
    print(f"  Cochran Q = {qrow['statistic']:.1f}, p = {fmt_p(qrow['p_value'])}")
    headline("RQ1", "Cochran Q p-value (all items)", qrow["p_value"], note="rates differ across pairs if small")

    # plot
    plt = setup_style()
    scopes_plot = [s for s, _ in scopes if s in set(rates["scope"])]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.26
    colors = [BLUE, ORANGE, GREY]
    for j, p in enumerate(PAIRS):
        sub = rates[rates["pair"] == PAIR_LABEL[p]].set_index("scope").loc[scopes_plot]
        x = np.arange(len(scopes_plot)) + (j - 1) * w
        ax.bar(x, 100 * sub["rate"], width=w * 0.92, color=colors[j], label=PAIR_LABEL[p])
        ax.errorbar(x, 100 * sub["rate"], yerr=[100 * (sub["rate"] - sub["rate_ci_lo"]),
                                                 100 * (sub["rate_ci_hi"] - sub["rate"])],
                    fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2)
    ax.set_xticks(np.arange(len(scopes_plot)))
    ax.set_xticklabels([f"{s}\n(n={int(rates[rates['scope']==s]['n_items'].iloc[0])})" for s in scopes_plot])
    ax.set_ylabel("Items where the pair disagrees (%)")
    ax.set_title("RQ1: share of items on which two critics differ by 2 or more points (95% Wilson intervals)")
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(0, max(100 * rates["rate_ci_hi"]) * 1.25)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_f(fig, "rq1_disagreement_rate_by_critic_pair.png")
    plt.close(fig)


# ============================================================================== RQ2
def pct_rank(x, ds):
    out = np.empty(len(x))
    for d in np.unique(ds):
        m = ds == d
        out[m] = (rankdata(x[m]) - 0.5) / m.sum()
    return out


def _corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def pooled_within(x, y, ds):
    return _corr(pct_rank(x, ds), pct_rank(y, ds))


def partial_within(x, y, z, ds):
    rx, ry, rz = pct_rank(x, ds), pct_rank(y, ds), pct_rank(z, ds)
    Z = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return _corr(ex, ey)


def spearman_plain(x, y):
    return _corr(rankdata(x), rankdata(y))


def boot_stat(fn, n, seed=SEED, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = fn(rng.integers(0, n, n))
        if v == v:
            vals.append(v)
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if vals else (np.nan, np.nan)


def rq2(df):
    print("\n=== RQ2: disagreement vs human quality score ===")
    y, ds = df["y"].values, df["dataset"].values
    z = df["score_mean"].values
    measures = [("Disagreement events (count)", "disagreement_count"),
                ("Largest score gap", "score_gap_max"),
                ("Mean pairwise score gap", "score_gap_mean"),
                ("Score variance", "score_variance")]
    rows = []
    for label, col in measures:
        x = df[col].values.astype(float)
        e = spearman_plain(x, y)
        lo, hi = boot_stat(lambda i: spearman_plain(x[i], y[i]), len(x))
        rows.append({"measure": label, "scope": "All items, plain Spearman", "n": len(x), "estimate": e,
                     "ci_lo": lo, "ci_hi": hi})
        e = pooled_within(x, y, ds)
        lo, hi = boot_stat(lambda i: pooled_within(x[i], y[i], ds[i]), len(x))
        rows.append({"measure": label, "scope": "Pooled, ranked within dataset", "n": len(x), "estimate": e,
                     "ci_lo": lo, "ci_hi": hi})
        e = partial_within(x, y, z, ds)
        lo, hi = boot_stat(lambda i: partial_within(x[i], y[i], z[i], ds[i]), len(x))
        rows.append({"measure": label, "scope": "Pooled, ranked within dataset, controlling for mean critic score",
                     "n": len(x), "estimate": e, "ci_lo": lo, "ci_hi": hi})
        for d in DATASET_LABEL:
            m = ds == d
            xs, ys = x[m], y[m]
            e = spearman_plain(xs, ys)
            lo, hi = boot_stat(lambda i: spearman_plain(xs[i], ys[i]), m.sum())
            rows.append({"measure": label, "scope": f"Spearman within {DATASET_LABEL[d]}", "n": int(m.sum()),
                         "estimate": e, "ci_lo": lo, "ci_hi": hi})
    # reference: the mean critic score itself
    e = pooled_within(z, y, ds)
    lo, hi = boot_stat(lambda i: pooled_within(z[i], y[i], ds[i]), len(z))
    rows.append({"measure": "Mean critic score (reference, not a disagreement measure)",
                 "scope": "Pooled, ranked within dataset", "n": len(z), "estimate": e, "ci_lo": lo, "ci_hi": hi})
    cor = pd.DataFrame(rows)
    save_t(cor, "rq2_disagreement_vs_human_score_correlations.csv")

    # mean human score by number of disagreement events, centred within dataset
    yc = y - pd.Series(y).groupby(ds).transform("mean").values
    cnt = df["disagreement_count"].values
    bins = np.minimum(cnt, 3).astype(int)
    brow = []
    for b in range(4):
        m = bins == b
        if m.sum() < 5:
            continue
        lo, hi = bootstrap_ci(yc[m], n_boot=N_BOOT)
        brow.append({"disagreement_events": "3 or more" if b == 3 else str(b), "n_items": int(m.sum()),
                     "mean_human_score_minus_dataset_mean": yc[m].mean(), "ci_lo": lo, "ci_hi": hi})
    byb = pd.DataFrame(brow)
    save_t(byb, "rq2_human_score_by_number_of_disagreement_events.csv")

    key = cor[cor["measure"] == "Disagreement events (count)"].set_index("scope")
    for s in ["Pooled, ranked within dataset", "Pooled, ranked within dataset, controlling for mean critic score"]:
        r = key.loc[s]
        print(f"  disagreement events, {s}: rho {r['estimate']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
        headline("RQ2", f"disagreement events vs human score, {s}", r["estimate"], r["ci_lo"], r["ci_hi"])
    r = cor[cor["measure"].str.startswith("Mean critic score")].iloc[0]
    print(f"  (reference) mean critic score, pooled within dataset: rho {r['estimate']:+.3f} "
          f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
    headline("RQ2", "mean critic score vs human score (reference)", r["estimate"], r["ci_lo"], r["ci_hi"])

    # plot
    plt = setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ypos = np.arange(len(measures))[::-1]
    for yy, (label, _) in zip(ypos, measures):
        for off, scope, col, mk in [(0.14, "Pooled, ranked within dataset", BLUE, "o"),
                                    (-0.14, "Pooled, ranked within dataset, controlling for mean critic score", ORANGE, "s")]:
            r = cor[(cor["measure"] == label) & (cor["scope"] == scope)].iloc[0]
            ax.plot([r["ci_lo"], r["ci_hi"]], [yy + off, yy + off], color=col, lw=1.4)
            ax.plot(r["estimate"], yy + off, mk, color=col, ms=5.5)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([m[0] for m in measures])
    ax.set_xlabel("Rank correlation with the human quality score")
    ax.set_title("RQ2: disagreement vs human score")
    ax.plot([], [], "o", color=BLUE, label="ranked within dataset")
    ax.plot([], [], "s", color=ORANGE, label="and controlling for mean critic score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    x = np.arange(len(byb))
    ax.errorbar(x, byb["mean_human_score_minus_dataset_mean"],
                yerr=[byb["mean_human_score_minus_dataset_mean"] - byb["ci_lo"],
                      byb["ci_hi"] - byb["mean_human_score_minus_dataset_mean"]],
                fmt="o-", color=BLUE, ms=5, lw=1.2, capsize=3)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n(n={n})" for a, n in zip(byb["disagreement_events"], byb["n_items"])])
    ax.set_xlabel("Disagreement events on the item")
    ax.set_ylabel("Human score minus dataset mean")
    ax.set_title("Quality by number of events")
    fig.tight_layout()
    save_f(fig, "rq2_disagreement_vs_human_score.png")
    plt.close(fig)


# ============================================================================== RQ3
def rq3(dims_df):
    print("\n=== RQ3: disagreement by quality dimension and task type ===")
    d = dims_df.copy()
    rows = []
    for tt in TASK_LABEL:
        for dim in DIM_LABEL:
            s = d[(d["task_type"] == tt) & (d["dimension"] == dim)]
            if len(s) == 0:
                continue
            k, n = int(s["D"].sum()), len(s)
            lo, hi = wilson(k, n)
            glo, ghi = bootstrap_ci(s["score_range"].values, n_boot=N_BOOT)
            rows.append({"task_type": TASK_LABEL[tt], "dimension": DIM_LABEL[dim], "n_items": n,
                         "items_with_disagreement": k, "rate": k / n, "rate_ci_lo": lo, "rate_ci_hi": hi,
                         "mean_score_range": s["score_range"].mean(), "range_ci_lo": glo, "range_ci_hi": ghi})
    for dim in DIM_LABEL:
        s = d[d["dimension"] == dim]
        k, n = int(s["D"].sum()), len(s)
        lo, hi = wilson(k, n)
        glo, ghi = bootstrap_ci(s["score_range"].values, n_boot=N_BOOT)
        rows.append({"task_type": "All task types", "dimension": DIM_LABEL[dim], "n_items": n,
                     "items_with_disagreement": k, "rate": k / n, "rate_ci_lo": lo, "rate_ci_hi": hi,
                     "mean_score_range": s["score_range"].mean(), "range_ci_lo": glo, "range_ci_hi": ghi})
    rates = pd.DataFrame(rows)
    save_t(rates, "rq3_disagreement_rate_by_dimension_and_task_type.csv")

    trows = []
    # (a) dimensions compared within a task type: same items, so paired tests
    for tt in TASK_LABEL:
        s = d[d["task_type"] == tt]
        dims = sorted(s["dimension"].unique())
        if len(dims) < 2:
            continue
        wideD = s.pivot_table(index="item_id", columns="dimension", values="D").dropna()
        wideR = s.pivot_table(index="item_id", columns="dimension", values="score_range").dropna()
        n = len(wideD)
        if n < 5:
            continue
        if len(dims) == 2:
            a, b = dims
            bb, cc, p = mcnemar_exact(wideD[a].values, wideD[b].values)
            trows.append({"comparison": f"Between dimensions within {TASK_LABEL[tt]}",
                          "test": f"Exact McNemar: {DIM_LABEL[a]} vs {DIM_LABEL[b]}", "n_items": n,
                          "statistic": np.nan, "p_value": p, "p_holm": np.nan,
                          "detail": f"only {DIM_LABEL[a]} disagrees {bb}, only {DIM_LABEL[b]} disagrees {cc}"})
            trows.append({"comparison": f"Between dimensions within {TASK_LABEL[tt]}",
                          "test": f"Wilcoxon on score range: {DIM_LABEL[a]} vs {DIM_LABEL[b]}", "n_items": n,
                          "statistic": np.nan, "p_value": safe_wilcoxon(wideR[a].values, wideR[b].values),
                          "p_holm": np.nan, "detail": ""})
        else:
            q, p = cochran_q(wideD[dims].values)
            trows.append({"comparison": f"Between dimensions within {TASK_LABEL[tt]}",
                          "test": "Cochran Q on 'any gap >= 2' (3 dimensions, paired)", "n_items": n,
                          "statistic": q, "p_value": p, "p_holm": np.nan, "detail": ""})
            fr = friedmanchisquare(*[wideR[c].values for c in dims])
            trows.append({"comparison": f"Between dimensions within {TASK_LABEL[tt]}",
                          "test": "Friedman on score range (3 dimensions, paired)", "n_items": n,
                          "statistic": float(fr.statistic), "p_value": float(fr.pvalue), "p_holm": np.nan,
                          "detail": ""})
            pw = []
            for a, b in itertools.combinations(dims, 2):
                bb, cc, p = mcnemar_exact(wideD[a].values, wideD[b].values)
                pw.append((a, b, bb, cc, p))
            for (a, b, bb, cc, p), ph in zip(pw, holm([x[4] for x in pw])):
                trows.append({"comparison": f"Between dimensions within {TASK_LABEL[tt]}",
                              "test": f"Exact McNemar: {DIM_LABEL[a]} vs {DIM_LABEL[b]}", "n_items": n,
                              "statistic": np.nan, "p_value": p, "p_holm": ph,
                              "detail": f"only first disagrees {bb}, only second disagrees {cc}"})
    # (b) task types compared within a dimension: different items, so independent-group tests
    for dim in DIM_LABEL:
        s = d[d["dimension"] == dim]
        groups = [g for _, g in s.groupby("task_type") if len(g) >= 5]
        if len(groups) < 2:
            continue
        table = np.array([[g["D"].sum(), len(g) - g["D"].sum()] for g in groups])
        if (table.sum(0) == 0).any():
            continue
        c2, p, dof, _ = chi2_contingency(table)
        names = ", ".join(TASK_LABEL[g["task_type"].iloc[0]] for g in groups)
        trows.append({"comparison": f"Between task types within {DIM_LABEL[dim]}",
                      "test": f"Chi-squared on 'any gap >= 2' ({names})", "n_items": int(table.sum()),
                      "statistic": float(c2), "p_value": float(p), "p_holm": np.nan, "detail": f"df={dof}"})
        kw = kruskal(*[g["score_range"].values for g in groups])
        trows.append({"comparison": f"Between task types within {DIM_LABEL[dim]}",
                      "test": "Kruskal-Wallis on score range", "n_items": int(table.sum()),
                      "statistic": float(kw.statistic), "p_value": float(kw.pvalue), "p_holm": np.nan, "detail": ""})
    tests = pd.DataFrame(trows)
    save_t(tests, "rq3_tests_dimension_and_task_type.csv")

    ov = rates[rates["task_type"] == "All task types"]
    for _, r in ov.iterrows():
        print(f"  {r['dimension']:20s} n={int(r['n_items']):4d}  disagree {100*r['rate']:5.1f}% "
              f"[{100*r['rate_ci_lo']:.1f}, {100*r['rate_ci_hi']:.1f}]  mean range {r['mean_score_range']:.2f}")
        headline("RQ3", f"disagreement rate, {r['dimension']} (all task types)", r["rate"], r["rate_ci_lo"],
                 r["rate_ci_hi"])
    for _, r in tests.iterrows():
        if r["test"].startswith(("Cochran", "Exact McNemar")) or r["test"].startswith("Chi-squared"):
            print(f"  {r['comparison']}: {r['test'][:70]}  p={fmt_p(r['p_value'])}"
                  + (f" (Holm {fmt_p(r['p_holm'])})" if r["p_holm"] == r["p_holm"] else ""))

    plt = setup_style()
    tts = [t for t in TASK_LABEL.values() if t in set(rates["task_type"])]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    w = 0.26
    colors = {"Factual accuracy": BLUE, "Logical consistency": ORANGE, "Completeness": GREY}
    for j, dim in enumerate(DIM_LABEL.values()):
        xs, vals, los, his = [], [], [], []
        for i, tt in enumerate(tts):
            r = rates[(rates["task_type"] == tt) & (rates["dimension"] == dim)]
            if len(r):
                r = r.iloc[0]
                xs.append(i + (j - 1) * w)
                vals.append(100 * r["rate"])
                los.append(100 * (r["rate"] - r["rate_ci_lo"]))
                his.append(100 * (r["rate_ci_hi"] - r["rate"]))
        ax.bar(xs, vals, width=w * 0.92, color=colors[dim], label=dim)
        ax.errorbar(xs, vals, yerr=[los, his], fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2)
    ax.set_xticks(np.arange(len(tts)))
    ax.set_xticklabels([f"{t}\n(n={int(rates[(rates['task_type']==t)]['n_items'].max())})" for t in tts])
    ax.set_ylabel("Items where critics differ by 2+ points (%)")
    ax.set_title("RQ3: disagreement rate by quality dimension and task type (95% Wilson intervals)")
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(0, 100 * rates["rate_ci_hi"].max() * 1.25)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_f(fig, "rq3_disagreement_by_dimension_and_task_type.png")
    plt.close(fig)


# ============================================================================== RQ4
def rq4(df):
    print("\n=== RQ4: learned arbitrator vs strong-model (GPT-5.6) judgement ===")
    y = df["y"].values
    ds = df["dataset"].values
    oof = cv_predictions(df, xgb_cond(FEATURE_NAMES))                       # (10 repeats, n)
    err_arb = np.abs(oof - y).mean(0)
    arb_round = np.clip(np.round(oof.mean(0)), 1, 10)
    err_arb_round = np.abs(arb_round - y)
    err_mean = np.abs(cv_predictions(df, linear_on("score_mean")) - y).mean(0)
    df = df.copy()
    df["err_arb"], df["err_arb_round"], df["err_mean"] = err_arb, err_arb_round, err_mean

    gpt_path = PROCESSED / "pointwise_terra_759.jsonl"
    have_gpt = gpt_path.exists()
    gpt_cost = np.nan
    if have_gpt:
        T = pd.DataFrame(load_jsonl(gpt_path)).drop_duplicates("item_id", keep="last").set_index("item_id")
        df["gpt"] = df["item_id"].map(T["terra_score"]).astype(float)
        df["gpt_cost"] = df["item_id"].map(T["cost_usd"]).astype(float)
        df["err_gpt"] = np.abs(df["gpt"] - df["y"])
        gpt_cost = float(df["gpt_cost"].mean())
        print(f"  GPT-5.6 results found for {int(df['gpt'].notna().sum())} items")
    else:
        print(f"  {gpt_path.name} not found: GPT-5.6 comparison skipped")

    scopes = [("MT-Bench (human)", df["dataset"] == "mt_bench_human", True),
              ("SummEval", df["dataset"] == "summeval", True),
              ("MT-Bench + SummEval", df["dataset"].isin(["mt_bench_human", "summeval"]), True),
              ("TruthfulQA", df["dataset"] == "truthfulqa", False),
              ("FActScore (labeled)", df["dataset"] == "factscore_labeled", False),
              ("All 759 items", df["dataset"].notna(), False)]
    rows, drows = [], []
    for name, mask, valid in scopes:
        s = df[mask.values]
        for lab, col in [("Mean of critics (linear fit)", "err_mean"), ("Learned arbitrator", "err_arb"),
                         ("Learned arbitrator, rounded to whole points", "err_arb_round")]:
            lo, hi = bootstrap_ci(s[col].values, n_boot=N_BOOT)
            rows.append({"scope": name, "system": lab, "n_items": len(s), "mae": s[col].mean(), "ci_lo": lo,
                         "ci_hi": hi, "usable_for_claims": True})
        if have_gpt and s["gpt"].notna().all():
            lo, hi = bootstrap_ci(s["err_gpt"].values, n_boot=N_BOOT)
            rows.append({"scope": name, "system": "GPT-5.6 (pointwise)", "n_items": len(s), "mae": s["err_gpt"].mean(),
                         "ci_lo": lo, "ci_hi": hi, "usable_for_claims": valid})
            for lab, col in [("Learned arbitrator", "err_arb"), ("Learned arbitrator, rounded", "err_arb_round")]:
                dlt, dlo, dhi = paired_bootstrap_diff(s[col].values, s["err_gpt"].values)
                drows.append({"scope": name, "comparison": f"{lab} minus GPT-5.6 (negative = arbitrator better)",
                              "n_items": len(s), "mae_difference": dlt, "ci_lo": dlo, "ci_hi": dhi,
                              "usable_for_claims": valid})
    mae = pd.DataFrame(rows)
    save_t(mae, "rq4_arbitrator_vs_gpt56_heldout_mae.csv")
    if drows:
        save_t(pd.DataFrame(drows), "rq4_arbitrator_vs_gpt56_paired_differences.csv")

    crit = sum(measured_critic_costs().values())
    cost = pd.DataFrame([
        {"system": "Critics + learned arbitrator (no escalation)", "cost_per_item_usd": crit,
         "note": "arbitrator runs locally; critic cost measured from API tokens on the arena run"},
        {"system": "Critics + GPT-5.6 on every item", "cost_per_item_usd": crit + gpt_cost,
         "note": "GPT-5.6 mean cost per call from the 759-item run" if have_gpt else "GPT-5.6 file missing"},
        {"system": "GPT-5.6 call alone", "cost_per_item_usd": gpt_cost, "note": ""},
        {"system": "Claude Haiku 4.5 critic alone", "cost_per_item_usd": measured_critic_costs()["critic_b"],
         "note": "for reference"}])
    save_t(cost, "rq4_cost_per_item.csv")

    def g(scope, system):
        r = mae[(mae["scope"] == scope) & (mae["system"] == system)]
        return r.iloc[0] if len(r) else None

    for scope in ["All 759 items", "MT-Bench + SummEval"]:
        for system in ["Mean of critics (linear fit)", "Learned arbitrator", "GPT-5.6 (pointwise)"]:
            r = g(scope, system)
            if r is not None:
                print(f"  {scope:22s} {system:34s} MAE {r['mae']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
                headline("RQ4", f"MAE, {system}, {scope}", r["mae"], r["ci_lo"], r["ci_hi"])
    if drows:
        dd = pd.DataFrame(drows)
        r = dd[(dd["scope"] == "MT-Bench + SummEval") & (dd["comparison"].str.startswith("Learned arbitrator minus"))]
        if len(r):
            r = r.iloc[0]
            print(f"  paired, arbitrator minus GPT-5.6 (MT-Bench + SummEval): {r['mae_difference']:+.3f} "
                  f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
            headline("RQ4", "paired MAE, arbitrator minus GPT-5.6 (MT-Bench + SummEval)", r["mae_difference"],
                     r["ci_lo"], r["ci_hi"], "negative = arbitrator better")
    print(f"  cost per item: critics+arbitrator ${crit:.5f}"
          + (f", critics+GPT-5.6 ${crit + gpt_cost:.5f}" if have_gpt else ""))
    headline("RQ4", "cost per item, critics + arbitrator (USD)", crit)
    if have_gpt:
        headline("RQ4", "cost per item, critics + GPT-5.6 (USD)", crit + gpt_cost)
    print("  NOTE: GPT-5.6 rows for TruthfulQA, FActScore and 'All 759' are not valid for claims (contaminated input).")

    # plot
    plt = setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [2.3, 1]})
    ax = axes[0]
    order = ["MT-Bench (human)", "SummEval", "TruthfulQA", "FActScore (labeled)", "All 759 items"]
    systems = [("Mean of critics (linear fit)", GREY), ("Learned arbitrator", BLUE), ("GPT-5.6 (pointwise)", ORANGE)]
    w = 0.26
    for j, (sysname, col) in enumerate(systems):
        xs, v, lo, hi = [], [], [], []
        for i, sc in enumerate(order):
            r = g(sc, sysname)
            if r is None or not r["usable_for_claims"]:
                continue
            xs.append(i + (j - 1) * w)
            v.append(r["mae"])
            lo.append(r["mae"] - r["ci_lo"])
            hi.append(r["ci_hi"] - r["mae"])
        ax.bar(xs, v, width=w * 0.92, color=col, label=sysname)
        ax.errorbar(xs, v, yerr=[lo, hi], fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{s}\n(n={int(g(s, 'Learned arbitrator')['n_items'])})" for s in order])
    ax.set_ylabel("MAE vs human score (1-10), lower is better")
    ax.set_title("RQ4: held-out error (GPT-5.6 shown only where valid)")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.set_ylim(0, 4.0)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    labs = ["Critics +\narbitrator", "Critics +\nGPT-5.6 on all"]
    vals = [crit, crit + gpt_cost if have_gpt else np.nan]
    ax.bar(range(2), vals, color=[BLUE, ORANGE], width=0.55)
    for i, v in enumerate(vals):
        if v == v:
            ax.text(i, v, f"${v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(2))
    ax.set_xticklabels(labs)
    ax.set_ylabel("USD per item")
    ax.set_title("Cost per item")
    ax.set_ylim(0, np.nanmax(vals) * 1.2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_f(fig, "rq4_arbitrator_vs_gpt56_error_and_cost.png")
    plt.close(fig)


# ============================================================================== RQ5
def rq5(df, by_id):
    print("\n=== RQ5: hallucination tracer ===")
    rows = []
    for _, r in df.iterrows():
        h = by_id[r["item_id"]]["hallucination_trace"]
        n_orig = len(h["origin_sentences"])
        deps = h["dependent_sentences"]
        per_origin = {}
        for dsent in deps:
            per_origin[dsent["depends_on_origin_index"]] = per_origin.get(dsent["depends_on_origin_index"], 0) + 1
        rows.append({"item_id": r["item_id"], "dataset": r["dataset"], "task_type": r["task_type"],
                     "traced": bool(h["has_hallucination"]), "n_origins": n_orig, "n_dependents": len(deps),
                     "propagation_depth": h["propagation_depth"],
                     "total_affected": h["total_affected_sentences"],
                     "mean_dependents_per_origin": (len(deps) / n_orig) if n_orig else np.nan})
    T = pd.DataFrame(rows)
    tr = T[T["traced"]]
    print(f"  items with a traced hallucination: {len(tr)} of {len(T)}")

    srows = []
    for name, s_all in [("All datasets", T)] + [(DATASET_LABEL[d], T[T["dataset"] == d]) for d in DATASET_LABEL]:
        s = s_all[s_all["traced"]]
        if len(s) == 0:
            srows.append({"scope": name, "items": len(s_all), "items_with_traced_hallucination": 0})
            continue
        row = {"scope": name, "items": len(s_all), "items_with_traced_hallucination": len(s),
               "share_traced": len(s) / len(s_all)}
        for col, lab in [("n_origins", "origin_sentences_per_traced_item"),
                         ("n_dependents", "dependent_sentences_per_traced_item"),
                         ("propagation_depth", "propagation_depth"),
                         ("total_affected", "affected_sentences_per_traced_item")]:
            lo, hi = bootstrap_ci(s[col].values, n_boot=N_BOOT)
            row[f"{lab}_mean"], row[f"{lab}_ci_lo"], row[f"{lab}_ci_hi"] = s[col].mean(), lo, hi
        row["share_of_traced_items_with_any_dependent"] = float((s["n_dependents"] > 0).mean())
        row["max_propagation_depth"] = int(s["propagation_depth"].max())
        srows.append(row)
    summ = pd.DataFrame(srows)
    save_t(summ, "rq5_propagation_depth_summary.csv")
    save_t(T, "rq5_tracer_per_item.csv")
    a = summ[summ["scope"] == "All datasets"].iloc[0]
    print(f"  mean propagation depth over traced items: {a['propagation_depth_mean']:.2f} "
          f"[{a['propagation_depth_ci_lo']:.2f}, {a['propagation_depth_ci_hi']:.2f}]; "
          f"{100*a['share_of_traced_items_with_any_dependent']:.0f}% of traced items have any dependent sentence; "
          f"max depth {a['max_propagation_depth']}")
    headline("RQ5", "mean propagation depth (traced items)", a["propagation_depth_mean"],
             a["propagation_depth_ci_lo"], a["propagation_depth_ci_hi"])
    headline("RQ5", "mean affected sentences per traced item", a["affected_sentences_per_traced_item_mean"],
             a["affected_sentences_per_traced_item_ci_lo"], a["affected_sentences_per_traced_item_ci_hi"])

    # origin validation, read from the tables tracer_origin_validation.py wrote earlier
    vt = TABLES / "tracer_origin_validation_vs_factscore_unsupported_sentences.csv"
    dt = TABLES / "tracer_dependent_sentences_vs_factscore_base_rate.csv"
    val = None
    if vt.exists():
        val = pd.read_csv(vt)
        save_t(val, "rq5_origin_validation_vs_factscore.csv")
        for _, r in val[val["method"] == "tracer"].iterrows():
            print(f"  tracer origin {r['metric']}: {r['value']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
            headline("RQ5", f"tracer origin {r['metric']} vs FActScore gold", r["value"], r["ci_lo"], r["ci_hi"])
        for m in ["random, same count", "every sentence"]:
            r = val[(val["method"] == m) & (val["metric"] == "precision")]
            if len(r):
                print(f"  {m} precision: {r.iloc[0]['value']:.3f}")
                headline("RQ5", f"{m} precision (baseline)", r.iloc[0]["value"], r.iloc[0]["ci_lo"], r.iloc[0]["ci_hi"])
    else:
        print(f"  {vt.name} not found: run FinalScripts/tracer_origin_validation.py first for the validation part")
    if dt.exists():
        d = pd.read_csv(dt).iloc[0]
        print(f"  dependent sentences that are gold-unsupported: {100*d['share']:.1f}% "
              f"[{100*d['ci_lo']:.1f}, {100*d['ci_hi']:.1f}] vs base rate {100*d['base_rate_of_unsupported_sentences']:.1f}%")
        headline("RQ5", "dependent sentences that are gold-unsupported", d["share"], d["ci_lo"], d["ci_hi"],
                 f"base rate {d['base_rate_of_unsupported_sentences']:.3f}")
        save_t(pd.read_csv(dt), "rq5_dependent_sentences_vs_base_rate.csv")

    plt = setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    ax = axes[0]
    if val is not None:
        methods = [("tracer", BLUE, "Tracer"), ("random, same count", GREY, "Random,\nsame count"),
                   ("every sentence", LIGHTGREY, "Every\nsentence"), ("first sentence only", ORANGE, "First\nsentence")]
        w = 0.36
        for j, metric in enumerate(["precision", "recall"]):
            for i, (mk, col, lab) in enumerate(methods):
                r = val[(val["method"] == mk) & (val["metric"] == metric)]
                if not len(r):
                    continue
                r = r.iloc[0]
                x = i + (j - 0.5) * w
                ax.bar(x, r["value"], width=w * 0.92, color=col, hatch="" if metric == "precision" else "//",
                       edgecolor="white" if metric == "precision" else "#444444", lw=0.5)
                ax.errorbar(x, r["value"], yerr=[[r["value"] - r["ci_lo"]], [r["ci_hi"] - r["value"]]],
                            fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m[2] for m in methods])
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=GREY, label="precision (solid)"),
                           Patch(facecolor="white", edgecolor="#444444", hatch="//", label="recall (hatched)")],
                  loc="upper right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Share")
        ax.set_title("RQ5: origin sentences vs FActScore (174 items)")
        ax.grid(axis="x", visible=False)
    else:
        ax.text(0.5, 0.5, "validation tables not found", ha="center", transform=ax.transAxes)
    ax = axes[1]
    depth = tr["propagation_depth"].clip(upper=6)
    counts = depth.value_counts().reindex(range(0, 7), fill_value=0)
    ax.bar(counts.index, counts.values, color=BLUE, width=0.7)
    ax.set_xticks(range(0, 7))
    ax.set_xticklabels(["0", "1", "2", "3", "4", "5", "6+"])
    ax.set_xlabel("Propagation depth of the item (longest dependency chain)")
    ax.set_ylabel("Items")
    ax.set_title(f"Propagation depth, {len(tr)} items with a traced hallucination")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_f(fig, "rq5_tracer_origin_accuracy_and_propagation.png")
    plt.close(fig)


# ============================================================================== main
def main():
    df, pairs_df, dims_df, by_id = build_tables()
    print(f"[rq] {len(df)} labelled items, {N_FEATURES} label-free features, outputs in {RQ_DIR.relative_to(ROOT)}")
    rq1(pairs_df)
    rq2(df)
    rq3(dims_df)
    rq4(df)
    rq5(df, by_id)
    save_t(pd.DataFrame(HEADLINE), "rq_headline_numbers.csv")


if __name__ == "__main__":
    main()