"""
scripts/compute_zheng_agreement.py

Computes ARBITER's pairwise agreement with human preference on the arena
dataset, using the same S1/S2 methodology Zheng et al. (2023, "Judging
LLM-as-a-Judge with MT-Bench and Chatbot Arena") report in their Table 6 --
plus an honest breakdown that Zheng et al.'s flat metric doesn't need,
because ARBITER's scores are an integer 1-10 scale and ties are common.

Reads run_results_arena.csv (needs: pair_id, side, human_vote_outcome,
verdict_quality_score, verdict_adjudicated_by, task_type). No rerun of the
pipeline is needed -- this is pure post-hoc analysis, zero API cost.

Writes:
  results/analysis/zheng_agreement_summary.csv
  results/analysis/zheng_agreement_per_pair.csv
  results/analysis/zheng_agreement_vs_reference.png
  results/analysis/zheng_agreement_by_gap.png

Usage:
  python -m scripts.compute_zheng_agreement
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_PATH = Path("results/processed/run_results_arena.csv")
OUT_DIR = Path("results/analysis")

TIE_LABELS = {"tie", "tie_bothbad"}

# Zheng et al. (2023), Table 6, Chatbot Arena, GPT-4 pairwise ("G4") vs Human ("H")
ZHENG_S1 = 64.0
ZHENG_S2 = 87.0
GAP_BINS = [(0, 1, "gap=0 (tie)"), (1, 2, "gap=1"), (2, 3, "gap=2"), (3, 5, "gap=3-4"), (5, 100, "gap>=5")]


def load_pairs(path: Path):
    pairs = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs[row["pair_id"]][row["side"]] = row
    return pairs


def human_pref_of(row_a):
    ho = row_a["human_vote_outcome"]
    if ho in TIE_LABELS:
        return "tie"
    if ho == "win":
        return "a"
    if ho == "loss":
        return "b"
    return None


def arbiter_pref_of(row_a, row_b):
    score_a, score_b = float(row_a["verdict_quality_score"]), float(row_b["verdict_quality_score"])
    if score_a > score_b:
        return "a", score_a - score_b
    if score_b > score_a:
        return "b", score_a - score_b
    return "tie", 0.0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(INPUT_PATH)

    per_pair_rows = []
    s1_total = s1_agree = 0
    s2_total = s2_agree = 0
    correct = arbiter_tied = wrong = 0
    gap_bucket = defaultdict(lambda: {"n": 0, "agree": 0})

    for pid, sides in pairs.items():
        if "a" not in sides or "b" not in sides:
            continue
        a, b = sides["a"], sides["b"]
        human_pref = human_pref_of(a)
        if human_pref is None:
            continue
        arbiter_pref, signed_gap = arbiter_pref_of(a, b)
        gap = abs(signed_gap)

        s1_total += 1
        s1_ok = arbiter_pref == human_pref
        if s1_ok:
            s1_agree += 1

        outcome = None
        if human_pref != "tie":
            s2_total += 1
            s2_ok = arbiter_pref == human_pref
            if s2_ok:
                s2_agree += 1

            if arbiter_pref == "tie":
                arbiter_tied += 1
                outcome = "arbiter_tied"
            elif s2_ok:
                correct += 1
                outcome = "correct"
            else:
                wrong += 1
                outcome = "wrong"

            for lo, hi, label in GAP_BINS:
                if lo <= gap < hi:
                    gap_bucket[label]["n"] += 1
                    if s2_ok:
                        gap_bucket[label]["agree"] += 1
                    break

        per_pair_rows.append({
            "pair_id": pid, "task_type": a["task_type"],
            "human_pref": human_pref, "arbiter_pref": arbiter_pref,
            "score_a": a["verdict_quality_score"], "score_b": b["verdict_quality_score"],
            "abs_gap": gap, "outcome": outcome or "tie_vote_excluded_from_s2",
            "method_a": a["verdict_adjudicated_by"], "method_b": b["verdict_adjudicated_by"],
        })

    decided = correct + wrong
    decisive_accuracy = 100 * correct / decided if decided else float("nan")

    # ---- per-pair CSV ----
    per_pair_path = OUT_DIR / "zheng_agreement_per_pair.csv"
    with open(per_pair_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_pair_rows)

    # ---- summary CSV ----
    summary_path = OUT_DIR / "zheng_agreement_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["complete_pairs_s1", s1_total])
        writer.writerow(["s1_agreement_pct", round(100 * s1_agree / s1_total, 2)])
        writer.writerow(["zheng_s1_reference_pct", ZHENG_S1])
        writer.writerow(["non_tie_pairs_s2", s2_total])
        writer.writerow(["s2_agreement_pct", round(100 * s2_agree / s2_total, 2)])
        writer.writerow(["zheng_s2_reference_pct", ZHENG_S2])
        writer.writerow(["s2_correct", correct])
        writer.writerow(["s2_arbiter_tied_no_verdict", arbiter_tied])
        writer.writerow(["s2_wrong", wrong])
        writer.writerow(["s2_arbiter_tie_rate_pct", round(100 * arbiter_tied / s2_total, 2)])
        writer.writerow(["decisive_accuracy_pct_excl_ties", round(decisive_accuracy, 2)])
        for lo, hi, label in GAP_BINS:
            d = gap_bucket[label]
            pct = round(100 * d["agree"] / d["n"], 2) if d["n"] else None
            writer.writerow([f"gap_bin_{label}_n", d["n"]])
            writer.writerow([f"gap_bin_{label}_agreement_pct", pct])

    print(f"[zheng_agreement] S1: {s1_agree}/{s1_total} = {100*s1_agree/s1_total:.1f}%  (Zheng ref: {ZHENG_S1}%)")
    print(f"[zheng_agreement] S2: {s2_agree}/{s2_total} = {100*s2_agree/s2_total:.1f}%  (Zheng ref: {ZHENG_S2}%)")
    print(f"[zheng_agreement] breakdown: correct={correct} tied={arbiter_tied} wrong={wrong}")
    print(f"[zheng_agreement] decisive accuracy (excl. arbiter ties): {decisive_accuracy:.1f}% (n={decided})")
    print(f"[zheng_agreement] wrote {summary_path}")
    print(f"[zheng_agreement] wrote {per_pair_path}")

    # ---- plot 1: ARBITER vs Zheng et al. reference bars ----
    plt.figure(figsize=(7, 4.5))
    labels = ["S1\n(with ties)", "S2\n(non-tie)", "Decisive\n(excl. arbiter ties)"]
    arbiter_vals = [100 * s1_agree / s1_total, 100 * s2_agree / s2_total, decisive_accuracy]
    zheng_vals = [ZHENG_S1, ZHENG_S2, None]
    x = range(len(labels))
    width = 0.35
    plt.bar([i - width / 2 for i in x], arbiter_vals, width, label="ARBITER", color="#4C72B0")
    plt.bar([i + width / 2 for i, v in zip(x, zheng_vals) if v is not None],
             [v for v in zheng_vals if v is not None], width,
             label="Zheng et al. (G4-Pair vs Human)", color="#C44E52")
    plt.axhline(50, color="gray", linestyle=":", linewidth=1, label="chance (non-tie)")
    plt.xticks(list(x), labels)
    plt.ylabel("Agreement (%)")
    plt.title("ARBITER pairwise agreement vs. Zheng et al. reference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "zheng_agreement_vs_reference.png", dpi=150)
    plt.close()

    # ---- plot 2: agreement by score gap ----
    plt.figure(figsize=(7, 4.5))
    gap_labels = [label for _, _, label in GAP_BINS]
    gap_pcts = [100 * gap_bucket[l]["agree"] / gap_bucket[l]["n"] if gap_bucket[l]["n"] else 0 for l in gap_labels]
    gap_ns = [gap_bucket[l]["n"] for l in gap_labels]
    bars = plt.bar(gap_labels, gap_pcts, color="#55A868")
    for bar, n in zip(bars, gap_ns):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"n={n}", ha="center", fontsize=9)
    plt.axhline(50, color="gray", linestyle=":", linewidth=1, label="chance")
    plt.ylabel("Agreement (%)")
    plt.xlabel("|score_a - score_b|")
    plt.title("Agreement rises with score gap (cf. Zheng et al. Fig. 2)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "zheng_agreement_by_gap.png", dpi=150)
    plt.close()

    print(f"[zheng_agreement] wrote plots to {OUT_DIR}")


if __name__ == "__main__":
    main()