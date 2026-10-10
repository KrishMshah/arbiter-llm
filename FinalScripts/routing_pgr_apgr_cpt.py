"""
FinalScripts/routing_pgr_apgr_cpt.py      (to-do item 11, FREE)

RouteLLM-style routing metrics (PGR, APGR, CPT; RouteLLM section 3.2) on the 759 labelled
items, using the item-10 results (results/processed/pointwise_terra_759.jsonl). No model calls.

Systems (same definitions as the arena analysis)
  weak   = out-of-fold arbitrator score, rounded to 1-10
  strong = gpt-5.6-terra adjudicator score on every item
  router = terra only on the items the router sends up; the arbitrator's score on the rest.
           The router sends up the LEAST confident items first (the deployed rule is
           confidence < 0.75, which sends 81.3% of items).
Quality measures ("higher is better" inside the code; MAE is shown as -MAE):
  MAE against the human 1-10 score (all items and each dataset), and the
  truthful/untruthful call accuracy on TruthfulQA (predicted score <= 5 means untruthful).
  Mean within-dataset Spearman is reported at the end points only (not decomposable per item).

PGR(c)  = (r(router at c) - r(weak)) / (r(strong) - r(weak)),  c = share of items sent to strong
APGR    = mean of PGR over c = 10%, 20%, ..., 100%   (RouteLLM's 10-point discretisation)
CPT(x)  = smallest c on a 5% grid with PGR >= x
Random router: sends a random share c of items up. For per-item-average measures its expected
PGR is exactly c, so expected APGR = 55% and CPT(x) = x%. That is the control line.

Honest handling of a small or negative gap
  PGR divides by r(strong) - r(weak). If terra is not measurably better than the arbitrator on
  a measure, PGR is not meaningful there. The tables report, for every measure, the gap with its
  interval and the share of bootstrap resamples in which strong beats weak. PGR/APGR/CPT are
  filled in ONLY where the 95% interval of the gap (strong minus weak) lies above zero; on every
  other measure the row says "PGR not meaningful" and the PGR/APGR/CPT cells stay empty.
  Where they are filled, their intervals use only resamples where strong beats weak.
Intervals: bootstrap over items (1,000 resamples), 95% percentile.

Run from the repo root:  python FinalScripts/routing_pgr_apgr_cpt.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import *

IN_PATH = PROCESSED / "pointwise_terra_759.jsonl"
N_BOOT = 1000
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)               # share of items sent to strong
APGR_POINTS = np.round(np.arange(0.1, 1.0001, 0.1), 2)       # RouteLLM's 10 points
CPT_TARGETS = (0.5, 0.8)


# ------------------------------------------------------------------ measures
def build_measures(R):
    """Per-item quality contributions for weak and strong, per measure. Returns dict
    name -> (q_weak, q_strong, in_subset mask, sign for plotting, higher_is_better)."""
    w = R["weak"].values.astype(float)
    s = R["strong"].values.astype(float)
    h = R["human"].values.astype(float)
    out = {}
    all_mask = np.ones(len(R), bool)
    out["MAE vs human score, all 759 items"] = (-np.abs(w - h), -np.abs(s - h), all_mask, -1, False)
    for key, label in DATASET_LABEL.items():
        m = (R["dataset"] == key).values
        out[f"MAE vs human score, {label}"] = (-np.abs(w - h), -np.abs(s - h), m, -1, False)
    t = (R["dataset"] == "truthfulqa").values
    truth_untruthful = h < 5
    out["Truthful/untruthful accuracy, TruthfulQA"] = (
        ((w <= 5) == truth_untruthful).astype(float), ((s <= 5) == truth_untruthful).astype(float), t, 1, True)
    return out


def curve_stats(q_w, q_s, mask, conf, esc, idx):
    """Quality at every grid share (router ordered by ascending confidence), at the deployed
    point, and the end points, on the items `idx` (a bootstrap sample or arange)."""
    c, qw, qs, m, e = conf[idx], q_w[idx], q_s[idx], mask[idx], esc[idx]
    n_sub = m.sum()
    order = np.argsort(c, kind="stable")
    qw_o, qs_o, m_o = qw[order] * m[order], qs[order] * m[order], m[order]
    n = len(order)
    cs_s = np.concatenate([[0.0], np.cumsum(qs_o)])
    cs_w = np.concatenate([[0.0], np.cumsum(qw_o)])
    k = np.round(GRID * n).astype(int)
    r_grid = (cs_s[k] + (cs_w[-1] - cs_w[k])) / n_sub
    r_weak, r_strong = cs_w[-1] / n_sub, cs_s[-1] / n_sub
    r_dep = ((qs * m)[e].sum() + (qw * m)[~e].sum()) / n_sub
    return r_grid, r_weak, r_strong, r_dep, e.mean()


def pgr_from(r_grid, r_weak, r_strong):
    gap = r_strong - r_weak
    if not gap > 0:
        return None
    return (r_grid - r_weak) / gap


def summarise_pgr(r_grid, r_weak, r_strong, r_dep, dep_frac):
    pg = pgr_from(r_grid, r_weak, r_strong)
    if pg is None:
        return None
    apgr = np.mean([pg[np.where(GRID == c)[0][0]] for c in APGR_POINTS])
    cpt = {}
    for x in CPT_TARGETS:
        hit = np.where(pg >= x)[0]
        cpt[x] = GRID[hit[0]] if len(hit) else np.nan
    pgr_dep = (r_dep - r_weak) / (r_strong - r_weak)
    return {"pgr_dep": pgr_dep, "apgr": apgr, "cpt50": cpt[0.5], "cpt80": cpt[0.8], "pg": pg}


def pct(a, lo=2.5, hi=97.5):
    a = np.asarray([x for x in a if x == x], float)
    return (np.percentile(a, lo), np.percentile(a, hi)) if len(a) else (np.nan, np.nan)


# ------------------------------------------------------------------ main
def main():
    plt = setup_style()
    if not IN_PATH.exists():
        raise SystemExit(f"{IN_PATH} not found - run item 10 first")
    R = pd.DataFrame(load_jsonl(IN_PATH))
    n = len(R)
    R["weak"] = np.clip(np.round(R["oof_pred"]), 1, 10)
    R["strong"] = R["terra_score"].astype(float)
    conf, esc = R["confidence"].values.astype(float), R["would_escalate"].values.astype(bool)
    print(f"[routing] {n} items, router sends {100*esc.mean():.1f}% to GPT-5.6 at the deployed rule")

    # costs
    crit = sum(measured_critic_costs().values())
    terra_cost = R["cost_usd"].mean()
    cost_always = crit + terra_cost
    cost_dep = crit + R.loc[esc, "cost_usd"].sum() / n
    C = pd.DataFrame([{
        "critics_cost_per_item_usd": crit, "terra_cost_per_item_usd": terra_cost,
        "deployed_share_sent_to_terra_pct": 100 * esc.mean(),
        "cascade_cost_per_item_usd": cost_dep, "always_escalate_cost_per_item_usd": cost_always,
        "cascade_saving_vs_always_escalate_pct": 100 * (cost_always - cost_dep) / cost_always,
        "critics_only_cost_per_item_usd": crit}])
    save_table(C, "routing_cost_per_item_cascade_vs_always_escalate.csv")
    print(C.T.round(5).to_string(header=False))

    measures = build_measures(R)
    rng = np.random.default_rng(SEED)
    boot_idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    allidx = np.arange(n)

    end_rows, pgr_rows, curve_rows = [], [], []
    plot_data = {}
    for name, (q_w, q_s, mask, sign, hib) in measures.items():
        pt = curve_stats(q_w, q_s, mask, conf, esc, allidx)
        r_grid, r_w, r_s, r_dep, dep_frac = pt
        bs = [curve_stats(q_w, q_s, mask, conf, esc, ix) for ix in boot_idx]
        gaps = np.array([b[2] - b[1] for b in bs])
        better = float((gaps > 0).mean())
        glo, ghi = pct(gaps)
        dep_q = [b[3] for b in bs]
        dlo, dhi = pct(dep_q)
        end_rows.append({"measure": name, "weak": sign * r_w, "strong": sign * r_s,
                         "router_at_deployed_rule": sign * r_dep,
                         "router_ci_lo": sign * (dhi if sign < 0 else dlo), "router_ci_hi": sign * (dlo if sign < 0 else dhi),
                         "strong_minus_weak_quality_gap": r_s - r_w, "gap_ci_lo": glo, "gap_ci_hi": ghi,
                         "share_of_resamples_strong_beats_weak": better,
                         "gap_ci_excludes_zero": bool(glo > 0 or ghi < 0),
                         "higher_is_better": hib})

        # curve with bands
        grid_boot = np.array([b[0] for b in bs])
        lo, hi = np.percentile(grid_boot, 2.5, axis=0), np.percentile(grid_boot, 97.5, axis=0)
        random_line = r_w + GRID * (r_s - r_w)
        for j, c in enumerate(GRID):
            a, b_ = sign * lo[j], sign * hi[j]
            curve_rows.append({"measure": name, "share_sent_to_strong_pct": 100 * c,
                               "dollars_per_item": crit + c * terra_cost,
                               "router": sign * r_grid[j], "router_ci_lo": min(a, b_), "router_ci_hi": max(a, b_),
                               "random_router_expected": sign * random_line[j]})
        plot_data[name] = (sign, hib, r_grid, r_w, r_s, r_dep, dep_frac, lo, hi)

        # PGR / APGR / CPT: only where the strong-minus-weak gap is measurably above zero
        point = summarise_pgr(r_grid, r_w, r_s, r_dep, dep_frac)
        stats = [summarise_pgr(*b) for b in bs]
        valid = [s_ for s_ in stats if s_ is not None]
        vshare = len(valid) / N_BOOT
        row = {"measure": name, "deployed_share_sent_to_strong_pct": 100 * dep_frac,
               "share_of_resamples_with_strong_better_than_weak": vshare,
               "random_router_expected_apgr_pct": 100 * APGR_POINTS.mean(),
               "random_router_expected_cpt50_pct": 50.0, "random_router_expected_cpt80_pct": 80.0}
        if point is None or not glo > 0:
            row["status"] = "PGR not meaningful: GPT-5.6 does not measurably beat the arbitrator here (gap interval includes zero or is negative)"
        else:
            row["status"] = "defined"
            for key, scale in (("pgr_dep", 100), ("apgr", 100), ("cpt50", 100), ("cpt80", 100)):
                vals = [v[key] for v in valid]
                l, h_ = pct(vals)
                row[f"{key}_pct"] = scale * point[key]
                row[f"{key}_ci_lo"], row[f"{key}_ci_hi"] = scale * l, scale * h_
            diffs = [v["apgr"] - APGR_POINTS.mean() for v in valid]
            l, h_ = pct(diffs)
            row["apgr_minus_random_pct_points"] = 100 * (point["apgr"] - APGR_POINTS.mean())
            row["apgr_minus_random_ci_lo"], row["apgr_minus_random_ci_hi"] = 100 * l, 100 * h_
        pgr_rows.append(row)

    # Spearman at the end points only
    def mean_spearman(score, ds, hum):
        vals = []
        for d in DATASET_LABEL:
            m = ds == d
            vals.append(spearmanr(score[m], hum[m])[0])
        return float(np.nanmean(vals))
    ds, hum = R["dataset"].values, R["human"].values.astype(float)
    router_scores = np.where(esc, R["strong"].values, R["weak"].values)

    def sp_triplet(ix):
        return (mean_spearman(R["weak"].values[ix], ds[ix], hum[ix]),
                mean_spearman(R["strong"].values[ix], ds[ix], hum[ix]),
                mean_spearman(router_scores[ix], ds[ix], hum[ix]))
    p = sp_triplet(allidx)
    b = np.array([sp_triplet(ix) for ix in boot_idx[:300]])
    gl, gh = pct(b[:, 1] - b[:, 0])
    rl, rh = pct(b[:, 2])
    end_rows.append({"measure": "Mean within-dataset Spearman with the human score (4 datasets)", "weak": p[0], "strong": p[1],
                     "router_at_deployed_rule": p[2], "router_ci_lo": rl, "router_ci_hi": rh,
                     "strong_minus_weak_quality_gap": p[1] - p[0], "gap_ci_lo": gl, "gap_ci_hi": gh,
                     "share_of_resamples_strong_beats_weak": float((b[:, 1] > b[:, 0]).mean()),
                     "gap_ci_excludes_zero": bool(gl > 0 or gh < 0), "higher_is_better": True})

    E = pd.DataFrame(end_rows)
    save_table(E, "routing_weak_vs_strong_end_points_by_measure.csv")
    P = pd.DataFrame(pgr_rows)
    save_table(P, "routing_pgr_apgr_cpt_with_random_router.csv")
    save_table(pd.DataFrame(curve_rows), "routing_quality_vs_share_sent_to_strong_curves.csv")
    pd.set_option("display.width", 250)
    print("\n[end points] strong - weak gap in quality (positive = GPT-5.6 better), with 95% interval")
    print(E[["measure", "weak", "strong", "router_at_deployed_rule", "strong_minus_weak_quality_gap",
             "gap_ci_lo", "gap_ci_hi", "share_of_resamples_strong_beats_weak"]].round(3).to_string(index=False))
    cols = [c for c in ["measure", "status", "deployed_share_sent_to_strong_pct", "pgr_dep_pct", "pgr_dep_ci_lo", "pgr_dep_ci_hi",
                        "apgr_pct", "apgr_ci_lo", "apgr_ci_hi", "cpt50_pct", "cpt80_pct",
                        "apgr_minus_random_pct_points", "apgr_minus_random_ci_lo", "apgr_minus_random_ci_hi"] if c in P.columns]
    print("\n[PGR / APGR / CPT] (random router: APGR 55%, CPT(50)=50%, CPT(80)=80%)")
    print(P[cols].round(1).to_string(index=False))

    # ------------------------------------------------------------ plot
    panels = ["MAE vs human score, all 759 items", "MAE vs human score, FActScore (labeled)",
              "MAE vs human score, MT-Bench (human)", "MAE vs human score, SummEval",
              "MAE vs human score, TruthfulQA", "Truthful/untruthful accuracy, TruthfulQA"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ax, name in zip(axes.ravel(), panels):
        sign, hib, r_grid, r_w, r_s, r_dep, dep_frac, lo, hi = plot_data[name]
        x = 100 * GRID
        y = sign * r_grid
        a, b_ = sign * lo, sign * hi
        ax.fill_between(x, np.minimum(a, b_), np.maximum(a, b_), color=BLUE, alpha=0.13, lw=0)
        ax.plot(x, y, "-o", color=BLUE, ms=3, lw=1.4, label="Router (least confident first)")
        ax.plot([0, 100], [sign * r_w, sign * r_s], "--", color=GREY, lw=1.2, label="Random router (expected)")
        ax.plot(100 * dep_frac, sign * r_dep, "s", color=ORANGE, ms=7, zorder=5, label="Deployed rule (confidence < 0.75)")
        ax.set_xlim(-2, 102)
        ax.set_xlabel("Share of items sent to the strong judge (%)")
        ax.set_ylabel("MAE (lower is better)" if not hib else "Accuracy (higher is better)")
        ax.set_title(name.replace(", all 759 items", ", all 759 items").replace("Truthful/untruthful accuracy, TruthfulQA",
                     "Truthful / untruthful accuracy, TruthfulQA"), fontsize=10)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Quality vs share of items sent to GPT-5.6: confidence-ordered router vs random router "
                 "(shaded = 95% bootstrap interval over items)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "routing_quality_vs_share_sent_to_strong_judge_759_items.png")
    plt.close(fig)

    # compact two-panel figure for the paper: all items + FActScore (the six-panel figure stays in the repo)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, name, ttl in zip(axes, ["MAE vs human score, all 759 items", "MAE vs human score, FActScore (labeled)"],
                             ["All 759 items", "FActScore (174 items)"]):
        sign, hib, r_grid, r_w, r_s, r_dep, dep_frac, lo, hi = plot_data[name]
        x = 100 * GRID
        a, b_ = sign * lo, sign * hi
        ax.fill_between(x, np.minimum(a, b_), np.maximum(a, b_), color=BLUE, alpha=0.13, lw=0)
        ax.plot(x, sign * r_grid, "-o", color=BLUE, ms=2.5, lw=1.2, label="Router")
        ax.plot([0, 100], [sign * r_w, sign * r_s], "--", color=GREY, lw=1.1, label="Random router")
        ax.plot(100 * dep_frac, sign * r_dep, "s", color=ORANGE, ms=5.5, zorder=5, label="Deployed rule")
        ax.set_xlabel("Items sent to GPT-5.6 (%)", fontsize=8.5)
        ax.set_ylabel("MAE vs human score", fontsize=8.5)
        ax.set_title(ttl, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    save_fig(fig, "routing_quality_vs_share_sent_to_gpt56_paper_two_panel.png")
    plt.close(fig)

    # second plot: PGR curve for the measures where the gap is defined and measurable
    ok = [r for r in end_rows[:-1] if r["gap_ci_excludes_zero"] and r["strong_minus_weak_quality_gap"] > 0]
    if ok:
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        for r in ok:
            sign, hib, r_grid, r_w, r_s, *_ = plot_data[r["measure"]]
            pg = pgr_from(r_grid, r_w, r_s)
            ax.plot(100 * GRID, 100 * pg, "-o", ms=3, lw=1.3, label=r["measure"].replace("vs human score, ", ""))
        ax.plot([0, 100], [0, 100], "--", color=GREY, lw=1.2, label="Random router (expected)")
        ax.set_xlabel("Share of items sent to the strong judge (%)")
        ax.set_ylabel("Performance gap recovered, PGR (%)")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("PGR vs share sent to GPT-5.6, only where it measurably beats the arbitrator", fontsize=10.5)
        fig.tight_layout()
        save_fig(fig, "routing_performance_gap_recovered_where_strong_beats_weak.png")
        plt.close(fig)
    else:
        print("[routing] no measure where GPT-5.6 measurably beats the arbitrator: no PGR plot (this is itself a result)")


if __name__ == "__main__":
    main()