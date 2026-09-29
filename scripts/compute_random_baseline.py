"""
scripts/compute_random_baseline.py

RouteLLM-style random-router control (their Table 1-3's "Random" row), plus
a bootstrap 95% CI on ARBITER's own APGR -- so "-1.8% vs random" can be read
against ARBITER's own sampling uncertainty instead of as a precise number.

Random router: at each cost fraction c, sends c% of the 260 items to the
strong model at random, averaged over N_TRIALS draws per c.

Bootstrap: resamples the 85 non-tie pairs with replacement, N_BOOT times,
recomputing ARBITER's APGR each time (same cost-grid method as the random
router, for direct comparability) to get a range instead of one point.

Both are pure local recomputation on data you already have -- zero API cost.

Reads:  results/processed/run_results_arena.csv
        results/processed/run_results_arena_reprocessed.jsonl
        results/processed/strong_baseline_backfill.jsonl
        results/analysis/pgr_apgr_cpt_summary.csv
        results/analysis/pgr_apgr_cpt_curve.csv
Writes: results/analysis/random_baseline_curve.csv
        results/analysis/random_baseline_summary.csv
        results/analysis/pgr_vs_random.png

Usage: python -m scripts.compute_random_baseline
"""

import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.compute_pgr_apgr_cpt import (
    ARENA_CSV, REPROCESSED_JSONL, BACKFILL_JSONL, OUT_DIR,
    load_arena_csv, load_reprocessed, load_backfill, pid_side, s2_agreement,
    human_pref_of, pref_from_scores,
)

N_TRIALS = 1000   # random-router Monte Carlo draws per cost level
N_BOOT = 1000     # bootstrap resamples for ARBITER's own CI
COST_GRID = list(range(0, 101, 5))
CPT_TARGETS = [50, 80]
SEED = 42


def non_tie_pair_ids(arena_rows):
    pair_ids = sorted(set(pid for pid, _ in arena_rows.keys()))
    return [pid for pid in pair_ids
            if arena_rows.get((pid, "a")) and human_pref_of(arena_rows[(pid, "a")]) in ("a", "b")]


def s2_over_pairlist(pair_list, scores, arena_rows):
    """Same idea as s2_agreement, but takes a LIST of pair_ids that may
    contain repeats -- needed so a bootstrap resample counts a repeated
    pair more than once (a dict would silently dedupe it)."""
    total = agree = 0
    for pid in pair_list:
        a = scores.get((pid, "a"))
        b = scores.get((pid, "b"))
        if a is None or b is None:
            continue
        human_pref = human_pref_of(arena_rows[(pid, "a")])
        total += 1
        if pref_from_scores(a, b) == human_pref:
            agree += 1
    return 100 * agree / total if total else float("nan")


def apgr_for_pairlist(pair_list, ranked, weak_scores, strong_scores, arena_rows):
    """ARBITER's APGR evaluated on a given (possibly resampled) pair list.
    Escalates the lowest-confidence items first at each cost level -- same
    behavior as sweeping the real threshold, just parameterized by % cost
    so it lines up with the random router's own grid."""
    n = len(ranked)
    r_weak = s2_over_pairlist(pair_list, weak_scores, arena_rows)
    r_strong = s2_over_pairlist(pair_list, strong_scores, arena_rows)
    denom = r_strong - r_weak
    if not denom:
        return float("nan")

    xs, ys = [], []
    for c in COST_GRID:
        k = round(c / 100 * n)
        escalate_ids = set(iid for iid, _ in ranked[:k])
        hybrid = {}
        for iid, d in ranked:
            score = d["strong_score"] if iid in escalate_ids else d["weak_score"]
            if score is None:
                continue
            hybrid[pid_side(iid)] = score
        r_pct = s2_over_pairlist(pair_list, hybrid, arena_rows)
        xs.append(c)
        ys.append(100 * (r_pct - r_weak) / denom)

    return sum((xs[i] - xs[i - 1]) * (ys[i - 1] + ys[i]) / 2 for i in range(1, len(xs))) / 100.0


def bootstrap_apgr_ci(ranked, weak_scores, strong_scores, arena_rows, pair_pool, n_boot=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    point = apgr_for_pairlist(pair_pool, ranked, weak_scores, strong_scores, arena_rows)
    samples = []
    for _ in range(n_boot):
        resampled = rng.choices(pair_pool, k=len(pair_pool))
        val = apgr_for_pairlist(resampled, ranked, weak_scores, strong_scores, arena_rows)
        if val == val:  # skip nan
            samples.append(val)
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples)) - 1]
    return point, lo, hi, len(samples)


def main():
    random.seed(SEED)
    arena_rows = load_arena_csv(ARENA_CSV)
    items = load_reprocessed(REPROCESSED_JSONL)
    backfill = load_backfill(BACKFILL_JSONL)
    for iid, d in items.items():
        d["strong_score"] = d["strong_score_real"] if d["adjudicated"] else backfill.get(iid)

    ids = list(items.keys())
    n = len(ids)

    weak_scores = {pid_side(iid): d["weak_score"] for iid, d in items.items()}
    strong_scores = {pid_side(iid): d["strong_score"] for iid, d in items.items() if d["strong_score"] is not None}
    r_weak, _ = s2_agreement(weak_scores, arena_rows)
    r_strong, _ = s2_agreement(strong_scores, arena_rows)
    denom = r_strong - r_weak

    # ---- random-router Monte Carlo curve ----
    curve_rows = []
    for c in COST_GRID:
        k = round(c / 100 * n)
        trial_pgrs = []
        for _ in range(N_TRIALS):
            strong_ids = set(random.sample(ids, k))
            hybrid = {}
            for iid, d in items.items():
                score = d["strong_score"] if iid in strong_ids else d["weak_score"]
                if score is None:
                    continue
                hybrid[pid_side(iid)] = score
            r_pct, _ = s2_agreement(hybrid, arena_rows)
            pgr = 100 * (r_pct - r_weak) / denom if denom else float("nan")
            trial_pgrs.append(pgr)
        avg_pgr = sum(trial_pgrs) / len(trial_pgrs)
        curve_rows.append({"pct_calls_to_strong": c, "pgr_pct": round(avg_pgr, 2)})

    xs = [r["pct_calls_to_strong"] for r in curve_rows]
    ys = [r["pgr_pct"] for r in curve_rows]
    apgr_random = sum((xs[i] - xs[i - 1]) * (ys[i - 1] + ys[i]) / 2 for i in range(1, len(xs))) / 100.0
    cpt_random = {t: next((x for x, y in zip(xs, ys) if y >= t), None) for t in CPT_TARGETS}

    with open(OUT_DIR / "random_baseline_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pct_calls_to_strong", "pgr_pct"])
        w.writeheader()
        w.writerows(curve_rows)

    # ---- bootstrap CI on ARBITER's own APGR ----
    ranked = sorted(items.items(), key=lambda kv: kv[1]["confidence"])
    pair_pool = non_tie_pair_ids(arena_rows)
    arbiter_point, arbiter_lo, arbiter_hi, n_valid = bootstrap_apgr_ci(
        ranked, weak_scores, strong_scores, arena_rows, pair_pool
    )
    random_inside_ci = arbiter_lo <= apgr_random <= arbiter_hi

    print(f"[random] Random router APGR = {apgr_random:.2f}%")
    for t in CPT_TARGETS:
        print(f"[random] Random CPT({t}%) = {cpt_random[t]}")
    print(f"[boot] ARBITER APGR = {arbiter_point:.2f}%  (95% CI: {arbiter_lo:.2f}% - {arbiter_hi:.2f}%, n_boot={n_valid})")
    print(f"[boot] Random's APGR ({apgr_random:.2f}%) falls "
          f"{'INSIDE' if random_inside_ci else 'OUTSIDE'} ARBITER's 95% CI -- "
          f"{'not distinguishable from random at this sample size' if random_inside_ci else 'a real difference from random'}")

    arbiter_csv = {}
    summary_path = OUT_DIR / "pgr_apgr_cpt_summary.csv"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) == 2:
                    arbiter_csv[row[0]] = row[1]

    improvement = 100 * (arbiter_point - apgr_random) / apgr_random if apgr_random else float("nan")

    with open(OUT_DIR / "random_baseline_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["random_apgr_pct", round(apgr_random, 2)])
        for t in CPT_TARGETS:
            w.writerow([f"random_cpt_{t}pct", cpt_random[t]])
        w.writerow(["arbiter_apgr_pct_headline", arbiter_csv.get("apgr_pct")])
        w.writerow(["arbiter_apgr_pct_bootstrap_grid", round(arbiter_point, 2)])
        w.writerow(["arbiter_apgr_ci_lo", round(arbiter_lo, 2)])
        w.writerow(["arbiter_apgr_ci_hi", round(arbiter_hi, 2)])
        w.writerow(["random_apgr_inside_arbiter_ci", random_inside_ci])
        w.writerow(["arbiter_cpt_50pct", arbiter_csv.get("cpt_50pct_min_calls_to_strong_pct")])
        w.writerow(["arbiter_cpt_80pct", arbiter_csv.get("cpt_80pct_min_calls_to_strong_pct")])
        w.writerow(["improvement_over_random_pct", round(improvement, 2)])

    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, color="gray", linewidth=2, linestyle="--", label=f"Random router (APGR={apgr_random:.1f}%)")
    real_curve_path = OUT_DIR / "pgr_apgr_cpt_curve.csv"
    if real_curve_path.exists():
        with open(real_curve_path, encoding="utf-8") as f:
            real_rows = list(csv.DictReader(f))
        plt.plot([float(r["pct_calls_to_strong"]) for r in real_rows],
                  [float(r["pgr_pct"]) for r in real_rows],
                  color="#4C72B0", linewidth=2,
                  label=f"ARBITER (APGR={arbiter_point:.1f}%, 95% CI {arbiter_lo:.0f}-{arbiter_hi:.0f}%)")
    plt.xlabel("% of calls routed to strong model")
    plt.ylabel("PGR (%)")
    plt.title("ARBITER cascade vs. random router")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pgr_vs_random.png", dpi=150)
    plt.close()

    print(f"[random] wrote CSVs + plot to {OUT_DIR}")


if __name__ == "__main__":
    main()