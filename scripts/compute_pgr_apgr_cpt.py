"""
scripts/compute_pgr_apgr_cpt.py

Run on: 260-item Chatbot Arena cascade (run_results_arena_reprocessed.jsonl).
Performance r(x) = S2 agreement with human vote (same method as
compute_zheng_agreement.py).

weak   = ml_arbitrator_output.predicted_quality_score, rounded to the same
         1-10 int scale the real pipeline uses for an ML-only verdict
         (pipeline.py's _ml_only_verdict: max(1, min(10, round(x)))) --
         needed so weak isn't structurally tie-proof vs strong/router's
         native integer scale, which would bias S2 in weak's favor.
strong = adjudicator quality_score (223 real + 37 backfilled)
router = actual verdict.quality_score ARBITER produced

Formulae:
  x(tau)   = % of items routed to strong at confidence threshold tau
  PGR(tau) = (r(tau) - r(weak)) / (r(strong) - r(weak))
  APGR     = area under PGR(x) for x in [0, 100]  (trapezoidal)
  CPT(p%)  = min x such that PGR(x) >= p

Reads:  results/processed/run_results_arena.csv
        results/processed/run_results_arena_reprocessed.jsonl
        results/processed/strong_baseline_backfill.jsonl
Writes: results/analysis/pgr_apgr_cpt_summary.csv
        results/analysis/pgr_apgr_cpt_curve.csv
        results/analysis/pgr_curve.png
        results/analysis/apgr_shaded.png
        results/analysis/cpt_plot.png
        results/analysis/policy_accuracy_comparison.png

Usage: python -m scripts.compute_pgr_apgr_cpt
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARENA_CSV = Path("results/processed/run_results_arena.csv")
REPROCESSED_JSONL = Path("results/processed/run_results_arena_reprocessed.jsonl")
BACKFILL_JSONL = Path("results/processed/strong_baseline_backfill.jsonl")
OUT_DIR = Path("results/analysis")

TIE_LABELS = {"tie", "tie_bothbad"}
CPT_TARGETS = [25, 50, 75, 80]
DEPLOYED_TAU = 0.75


def load_arena_csv(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row["pair_id"], row["side"])] = row
    return rows


def load_reprocessed(path):
    items = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            raw_weak = d["ml_arbitrator_output"]["predicted_quality_score"]
            items[d["input_id"]] = {
                "weak_score": max(1, min(10, round(raw_weak))),
                "confidence": d["ml_arbitrator_output"]["arbitration_confidence"],
                "router_score": d["verdict"]["quality_score"],
                "adjudicated": d["verdict"]["adjudicated"],
                "strong_score_real": d["verdict"]["quality_score"] if d["verdict"]["adjudicated"] else None,
            }
    return items


def load_backfill(path):
    backfill = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    backfill[d["input_id"]] = d["strong_quality_score"]
    return backfill


def pid_side(input_id):
    pid, side = input_id.rsplit("_", 1)
    return pid, side


def human_pref_of(row_a):
    ho = row_a["human_vote_outcome"]
    if ho in TIE_LABELS:
        return "tie"
    if ho == "win":
        return "a"
    if ho == "loss":
        return "b"
    return None


def pref_from_scores(score_a, score_b):
    if score_a > score_b:
        return "a"
    if score_b > score_a:
        return "b"
    return "tie"


def s2_agreement(pairs_scores, arena_rows):
    by_pair = defaultdict(dict)
    for (pid, side), score in pairs_scores.items():
        by_pair[pid][side] = score

    total = agree = 0
    for pid, sides in by_pair.items():
        if "a" not in sides or "b" not in sides:
            continue
        row_a = arena_rows.get((pid, "a"))
        if row_a is None:
            continue
        human_pref = human_pref_of(row_a)
        if human_pref is None or human_pref == "tie":
            continue
        total += 1
        if pref_from_scores(sides["a"], sides["b"]) == human_pref:
            agree += 1

    pct = 100 * agree / total if total else float("nan")
    return pct, total


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arena_rows = load_arena_csv(ARENA_CSV)
    items = load_reprocessed(REPROCESSED_JSONL)
    backfill = load_backfill(BACKFILL_JSONL)

    missing = [iid for iid, d in items.items() if not d["adjudicated"] and iid not in backfill]
    if missing:
        print(f"[pgr] WARNING: {len(missing)} items still missing a strong score -- "
              f"run scripts.backfill_strong_baseline first.")

    for iid, d in items.items():
        d["strong_score"] = d["strong_score_real"] if d["adjudicated"] else backfill.get(iid)

    weak_scores = {pid_side(iid): d["weak_score"] for iid, d in items.items()}
    strong_scores = {pid_side(iid): d["strong_score"] for iid, d in items.items() if d["strong_score"] is not None}
    router_scores = {pid_side(iid): d["router_score"] for iid, d in items.items()}

    r_weak, n_weak = s2_agreement(weak_scores, arena_rows)
    r_strong, n_strong = s2_agreement(strong_scores, arena_rows)
    r_router, n_router = s2_agreement(router_scores, arena_rows)

    print(f"[pgr] r(weak)   = {r_weak:.2f}%  (n={n_weak})")
    print(f"[pgr] r(strong) = {r_strong:.2f}%  (n={n_strong}, {len(backfill)} backfilled)")
    print(f"[pgr] r(router) = {r_router:.2f}%  (n={n_router}) -- should match zheng_agreement_summary.csv S2 (54.12%)")

    if r_strong <= r_weak:
        print(f"[pgr] WARNING: r(strong) <= r(weak) -- PGR denominator is zero or negative. "
              f"Numbers below will be unstable/uninterpretable. Stop and investigate before using them.")

    confidences = sorted(set(d["confidence"] for d in items.values()))
    thresholds = sorted(set([0.0, 1.0, DEPLOYED_TAU] + confidences))

    curve_rows = []
    for tau in thresholds:
        hybrid = {}
        n_escalated = 0
        for iid, d in items.items():
            escalate = d["confidence"] < tau
            n_escalated += int(escalate)
            score = d["strong_score"] if escalate else d["weak_score"]
            if score is None:
                continue
            hybrid[pid_side(iid)] = score

        x_pct = 100 * n_escalated / len(items)
        r_pct, n_pairs = s2_agreement(hybrid, arena_rows)
        denom = r_strong - r_weak
        pgr = 100 * (r_pct - r_weak) / denom if denom else float("nan")
        curve_rows.append({"threshold": round(tau, 4), "pct_calls_to_strong": round(x_pct, 2),
                            "r_pct": round(r_pct, 2), "n_pairs": n_pairs, "pgr_pct": round(pgr, 2)})

    curve_rows.sort(key=lambda r: r["pct_calls_to_strong"])
    deployed_row = next(r for r in curve_rows if r["threshold"] == DEPLOYED_TAU)

    by_x = {r["pct_calls_to_strong"]: r["pgr_pct"] for r in curve_rows}
    xs = sorted(by_x)
    apgr = sum((xs[i] - xs[i - 1]) * (by_x[xs[i - 1]] + by_x[xs[i]]) / 2 for i in range(1, len(xs))) / 100.0

    cpt = {t: next((r["pct_calls_to_strong"] for r in curve_rows if r["pgr_pct"] >= t), None) for t in CPT_TARGETS}

    with open(OUT_DIR / "pgr_apgr_cpt_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "pct_calls_to_strong", "r_pct", "n_pairs", "pgr_pct"])
        w.writeheader()
        w.writerows(curve_rows)

    with open(OUT_DIR / "pgr_apgr_cpt_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["r_weak_pct", round(r_weak, 2)])
        w.writerow(["r_strong_pct", round(r_strong, 2)])
        w.writerow(["r_router_pct", round(r_router, 2)])
        w.writerow(["n_backfilled_strong", len(backfill)])
        w.writerow(["deployed_threshold", DEPLOYED_TAU])
        w.writerow(["deployed_pct_calls_to_strong", deployed_row["pct_calls_to_strong"]])
        w.writerow(["deployed_pgr_pct", deployed_row["pgr_pct"]])
        w.writerow(["apgr_pct", round(apgr, 2)])
        for t in CPT_TARGETS:
            w.writerow([f"cpt_{t}pct_min_calls_to_strong_pct", cpt[t]])

    print(f"\n[pgr] deployed (tau=0.75): {deployed_row['pct_calls_to_strong']:.1f}% to strong, PGR={deployed_row['pgr_pct']:.1f}%")
    print(f"[pgr] APGR = {apgr:.2f}%")
    for t in CPT_TARGETS:
        print(f"[pgr] CPT({t}%) = {cpt[t]}" if cpt[t] is not None else f"[pgr] CPT({t}%) = not reached")

    xs_curve = [r["pct_calls_to_strong"] for r in curve_rows]
    ys_curve = [r["pgr_pct"] for r in curve_rows]

    plt.figure(figsize=(7, 4.5))
    plt.plot(xs_curve, ys_curve, color="#4C72B0", linewidth=2, label="ARBITER cascade")
    plt.scatter([deployed_row["pct_calls_to_strong"]], [deployed_row["pgr_pct"]],
                color="#C44E52", zorder=5, label="deployed (\u03c4=0.75)")
    plt.xlabel("% of calls routed to strong model (adjudicator)")
    plt.ylabel("PGR (%)")
    plt.title("Performance Gap Recovered (PGR) vs. escalation rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pgr_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(xs_curve, ys_curve, color="#4C72B0", linewidth=2)
    plt.fill_between(xs_curve, ys_curve, 0, color="#4C72B0", alpha=0.2)
    plt.text(0.05, 0.92, f"APGR = {apgr:.2f}%", transform=plt.gca().transAxes,
              fontsize=11, fontweight="bold", va="top")
    plt.xlabel("% of calls routed to strong model (adjudicator)")
    plt.ylabel("PGR (%)")
    plt.title("Average PGR (APGR) -- area under the PGR curve")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "apgr_shaded.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(xs_curve, ys_curve, color="#4C72B0", linewidth=2)
    for t in CPT_TARGETS:
        plt.axhline(t, color="gray", linestyle=":", linewidth=0.8)
        if cpt[t] is not None:
            plt.axvline(cpt[t], color="gray", linestyle=":", linewidth=0.8)
            plt.scatter([cpt[t]], [t], color="#55A868", zorder=5)
            plt.annotate(f"CPT({t}%)={cpt[t]:.1f}%", (cpt[t], t),
                          textcoords="offset points", xytext=(6, -10), fontsize=9)
    plt.xlabel("% of calls routed to strong model (adjudicator)")
    plt.ylabel("PGR (%)")
    plt.title("Call-Performance Threshold (CPT)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cpt_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4.5))
    plt.bar(["Weak\n(ML only)", "Router\n(deployed)", "Strong\n(adjudicator only)"],
            [r_weak, r_router, r_strong], color=["#55A868", "#4C72B0", "#C44E52"])
    plt.axhline(50, color="gray", linestyle=":", linewidth=1, label="chance")
    plt.ylabel("S2 agreement with human vote (%)")
    plt.title("Weak / router / strong baselines (arena, non-tie pairs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "policy_accuracy_comparison.png", dpi=150)
    plt.close()

    print(f"[pgr] wrote CSVs + 4 plots to {OUT_DIR}")


if __name__ == "__main__":
    main()