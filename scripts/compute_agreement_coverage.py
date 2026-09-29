"""
scripts/compute_agreement_coverage.py

Agreement-vs-coverage curve, same framing as Trust-or-Escalate (Jung et al.,
ICLR 2025): abstain on low-confidence pairs, measure agreement on the rest.

Two confidence proxies compared:
  abs_gap            -- score separation between the two sides (coarse, item 7's signal)
  arbitration_confidence -- ML arbitrator's own continuous confidence (item 8's signal),
                            joined per pair as min(side_a_conf, side_b_conf)

S2 = non-tie human pairs only, correct = arbiter_pref == human_pref.
S1 = all pairs incl. ties, correct = arbiter_pref == human_pref (tie counts).

Sanity check (must match zheng_agreement_summary.csv):
  tau=-1 (no abstention): 54.12%, n=85
  tau=0 (exclude arbiter's own ties): 66.67%, n=69

Reads:  results/analysis/zheng_agreement_per_pair.csv
        results/processed/run_results_arena_reprocessed.jsonl  (optional, for after run)
Writes: results/coverage/agreement_coverage_curve.csv
        results/coverage/agreement_coverage_summary.csv
        results/coverage/agreement_vs_coverage.png
        results/coverage/selective_eval_vs_trust_or_escalate.png
        results/coverage/coverage_key_thresholds.png
        results/coverage/confidence_proxy_comparison.png  (only if reprocessed jsonl found)

Usage: python -m scripts.compute_agreement_coverage
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PER_PAIR_CSV = Path("results/analysis/zheng_agreement_per_pair.csv")
REPROCESSED_JSONL = Path("results/processed/run_results_arena_reprocessed.jsonl")
OUT_DIR = Path("results/coverage")

TRUST_OR_ESCALATE_COVERAGE_PCT = 80.0
TRUST_OR_ESCALATE_AGREEMENT_PCT = 80.0
CHANCE_S2_PCT = 50.0

COLOR_MAIN = "#1f77b4"
COLOR_ALT = "#ff7f0e"
COLOR_REF = "#333333"
COLOR_CHANCE = "#999999"


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["abs_gap"] = float(r["abs_gap"])
    return rows


def load_pair_confidence(path):
    """Join in the ML arbitrator's own confidence, min over both sides of a pair."""
    if not path.exists():
        return None
    side_conf = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pid, side = d["input_id"].rsplit("_", 1)
            side_conf.setdefault(pid, {})[side] = d["ml_arbitrator_output"]["arbitration_confidence"]
    return {pid: min(sides["a"], sides["b"]) for pid, sides in side_conf.items() if "a" in sides and "b" in sides}


def build_curve(pairs, population_filter, is_correct_fn, n_total, value_key="abs_gap"):
    """Sweep abstention threshold tau on value_key; keep pairs with value > tau."""
    pool = [p for p in pairs if population_filter(p) and p.get(value_key) is not None]
    thresholds = [-1.0] + sorted(set(p[value_key] for p in pool))

    curve = []
    for tau in thresholds:
        covered = [p for p in pool if p[value_key] > tau]
        if not covered:
            continue
        n_correct = sum(1 for p in covered if is_correct_fn(p))
        curve.append({
            "threshold": tau,
            "n_covered": len(covered),
            "coverage_pct": round(100 * len(covered) / n_total, 2),
            "agreement_pct": round(100 * n_correct / len(covered), 2),
        })

    seen, deduped = set(), []
    for row in curve:
        if row["n_covered"] not in seen:
            seen.add(row["n_covered"])
            deduped.append(row)
    deduped.sort(key=lambda r: r["coverage_pct"])
    return deduped


def s2_is_correct(p):
    return p["outcome"] == "correct"


def s1_is_correct(p):
    return p["arbiter_pref"] == p["human_pref"]


def nearest_point(curve, target_coverage_pct):
    return min(curve, key=lambda r: abs(r["coverage_pct"] - target_coverage_pct))


def min_coverage_for_agreement(curve, target_agreement_pct):
    for row in sorted(curve, key=lambda r: -r["coverage_pct"]):
        if row["agreement_pct"] >= target_agreement_pct:
            return row
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(PER_PAIR_CSV)

    n_s2_total = sum(1 for p in pairs if p["outcome"] != "tie_vote_excluded_from_s2")
    n_s1_total = len(pairs)
    s2_filter = lambda p: p["outcome"] != "tie_vote_excluded_from_s2"

    s2_curve = build_curve(pairs, s2_filter, s2_is_correct, n_s2_total, "abs_gap")
    s1_curve = build_curve(pairs, lambda p: True, s1_is_correct, n_s1_total, "abs_gap")

    # sanity check
    full_coverage = next(r for r in s2_curve if r["threshold"] == -1.0)
    decisive = next(r for r in s2_curve if r["threshold"] == 0.0)
    print(f"[check] full coverage: {full_coverage['agreement_pct']}% (n={full_coverage['n_covered']}) -- expect 54.12% / n=85")
    print(f"[check] decisive (tau=0): {decisive['agreement_pct']}% (n={decisive['n_covered']}) -- expect 66.67% / n=69")
    if abs(full_coverage["agreement_pct"] - 54.12) > 0.5 or abs(decisive["agreement_pct"] - 66.67) > 0.5:
        print("[check] MISMATCH -- verify input CSV before trusting plots.")

    at_80_coverage = nearest_point(s2_curve, TRUST_OR_ESCALATE_COVERAGE_PCT)
    max_cov_for_80_agree = min_coverage_for_agreement(s2_curve, TRUST_OR_ESCALATE_AGREEMENT_PCT)

    print(f"\n[coverage] agreement at ~80% coverage: {at_80_coverage['agreement_pct']}% (actual coverage {at_80_coverage['coverage_pct']}%)")
    if max_cov_for_80_agree:
        print(f"[coverage] max coverage holding >=80% agreement: {max_cov_for_80_agree['coverage_pct']}%")

    # ---- optional second proxy: arbitration_confidence ----
    pair_conf = load_pair_confidence(REPROCESSED_JSONL)
    conf_curve = None
    if pair_conf is not None:
        for p in pairs:
            p["pair_confidence"] = pair_conf.get(p["pair_id"])
        n_conf_pool = sum(1 for p in pairs if s2_filter(p) and p.get("pair_confidence") is not None)
        print(f"\n[conf-proxy] joined confidence for {n_conf_pool}/{n_s2_total} S2 pairs")
        conf_curve = build_curve(pairs, s2_filter, s2_is_correct, n_conf_pool, "pair_confidence")
    else:
        print(f"\n[conf-proxy] {REPROCESSED_JSONL} not found -- skipping confidence-based curve.")

    # ---- CSVs ----
    with open(OUT_DIR / "agreement_coverage_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["methodology", "threshold", "n_covered", "coverage_pct", "agreement_pct"])
        for r in s2_curve:
            w.writerow(["S2_abs_gap", r["threshold"], r["n_covered"], r["coverage_pct"], r["agreement_pct"]])
        for r in s1_curve:
            w.writerow(["S1_abs_gap", r["threshold"], r["n_covered"], r["coverage_pct"], r["agreement_pct"]])
        if conf_curve:
            for r in conf_curve:
                w.writerow(["S2_arbitration_confidence", r["threshold"], r["n_covered"], r["coverage_pct"], r["agreement_pct"]])

    summary = {
        "s2_full_coverage_agreement_pct": full_coverage["agreement_pct"],
        "s2_decisive_agreement_pct": decisive["agreement_pct"],
        "s2_decisive_coverage_pct": decisive["coverage_pct"],
        "s2_agreement_at_80pct_coverage": at_80_coverage["agreement_pct"],
        "s2_actual_coverage_near_80": at_80_coverage["coverage_pct"],
        "s2_max_coverage_holding_80pct_agreement": max_cov_for_80_agree["coverage_pct"] if max_cov_for_80_agree else "not_reached",
        "trust_or_escalate_reference_coverage_pct": TRUST_OR_ESCALATE_COVERAGE_PCT,
        "trust_or_escalate_reference_agreement_pct": TRUST_OR_ESCALATE_AGREEMENT_PCT,
    }
    if conf_curve:
        conf_at_80 = nearest_point(conf_curve, TRUST_OR_ESCALATE_COVERAGE_PCT)
        conf_max_80 = min_coverage_for_agreement(conf_curve, TRUST_OR_ESCALATE_AGREEMENT_PCT)
        summary["conf_agreement_at_80pct_coverage"] = conf_at_80["agreement_pct"]
        summary["conf_max_coverage_holding_80pct_agreement"] = conf_max_80["coverage_pct"] if conf_max_80 else "not_reached"

    with open(OUT_DIR / "agreement_coverage_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])

    # ---- plot 1 ----
    xs = [r["coverage_pct"] for r in s2_curve]
    ys = [r["agreement_pct"] for r in s2_curve]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, color=COLOR_MAIN, linewidth=1.5, marker="o", markersize=4)
    plt.axhline(CHANCE_S2_PCT, color=COLOR_CHANCE, linestyle="--", linewidth=1, label="chance")
    plt.xlabel("Coverage (% of pairs given a decisive verdict)")
    plt.ylabel("Agreement with human vote (%)")
    plt.title("Human agreement vs. coverage under selective abstention")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "agreement_vs_coverage.png", dpi=150)
    plt.close()

    # ---- plot 2 ----
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, color=COLOR_MAIN, linewidth=1.5, label="This work")
    plt.plot([TRUST_OR_ESCALATE_COVERAGE_PCT], [TRUST_OR_ESCALATE_AGREEMENT_PCT],
             color=COLOR_REF, marker="s", markersize=7, linestyle="none",
             label="Trust-or-Escalate (80% cov / 80% agree)")
    plt.axhline(CHANCE_S2_PCT, color=COLOR_CHANCE, linestyle="--", linewidth=1)
    plt.xlabel("Coverage (%)")
    plt.ylabel("Agreement with human vote (%)")
    plt.title("Selective evaluation vs. Trust-or-Escalate (Jung et al., ICLR 2025)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "selective_eval_vs_trust_or_escalate.png", dpi=150)
    plt.close()

    # ---- plot 3 ----
    labels = ["Full coverage", "Decisive\n(exclude ties)", "Max coverage >=80 agree", "Gap>=5 bin"]
    gap5 = next((r for r in s2_curve if r["threshold"] == 4.0), decisive)
    threshold_row = max_cov_for_80_agree if max_cov_for_80_agree else decisive
    values = [full_coverage["agreement_pct"], decisive["agreement_pct"], threshold_row["agreement_pct"], gap5["agreement_pct"]]
    covs = [full_coverage["coverage_pct"], decisive["coverage_pct"], threshold_row["coverage_pct"], gap5["coverage_pct"]]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, values, color=COLOR_MAIN)
    for bar, cov in zip(bars, covs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.1f}% ({cov:.0f}% cov)", ha="center", fontsize=8)
    plt.axhline(CHANCE_S2_PCT, color=COLOR_CHANCE, linestyle="--", linewidth=1, label="chance")
    plt.ylabel("Agreement with human vote (%)")
    plt.title("Agreement at key coverage levels")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "coverage_key_thresholds.png", dpi=150)
    plt.close()

    # ---- plot 4: confidence proxy comparison (only if joined) ----
    if conf_curve:
        cxs = [r["coverage_pct"] for r in conf_curve]
        cys = [r["agreement_pct"] for r in conf_curve]
        plt.figure(figsize=(7, 4.5))
        plt.plot(xs, ys, color=COLOR_MAIN, linewidth=1.5, marker="o", markersize=3, label="Proxy: score gap")
        plt.plot(cxs, cys, color=COLOR_ALT, linewidth=1.5, marker="o", markersize=3, label="Proxy: arbitrator confidence")
        plt.axhline(CHANCE_S2_PCT, color=COLOR_CHANCE, linestyle="--", linewidth=1)
        plt.xlabel("Coverage (%)")
        plt.ylabel("Agreement with human vote (%)")
        plt.title("Confidence proxy comparison: score gap vs. arbitrator confidence")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "confidence_proxy_comparison.png", dpi=150)
        plt.close()

    print(f"\n[coverage] wrote CSVs + plots to {OUT_DIR}")


if __name__ == "__main__":
    main()