"""
FinalScripts/arena_intervals_and_bias_checks.py

Puts confidence intervals on every existing Chatbot Arena result and adds the free
bias checks. Reads the stored 260-item arena run (130 pairs); no new model calls.

Systems compared (same definitions as scripts/compute_pgr_apgr_cpt.py):
  weak   = arbitrator predicted score rounded to 1-10
  router = ARBITER's actual verdict (arbitrator or adjudicator)
  strong = adjudicator score on every item (223 real + 37 backfilled)
S1 = all 130 pairs, a human tie counts as the label "tie".
S2 = the 85 pairs with a non-tie human vote.
Decisive accuracy = S2 restricted to pairs where the system itself did not tie.

Bias checks (pointwise scoring has no position bias by construction because each
answer is scored alone; the position check here only looks at which slot wins):
  verbosity: how often the longer answer is preferred, by humans and by ARBITER.

Run: python FinalScripts/arena_intervals_and_bias_checks.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, binomtest

from common import *

ZHENG = {"S1": 64.0, "S2": 87.0}     # Zheng et al. 2023 Table 6, Chatbot Arena, GPT-4 pairwise


def pref(a, b):
    return "a" if a > b else ("b" if b > a else "tie")


def main():
    plt = setup_style()
    sample = {it["item_id"]: it for it in load_jsonl(PROCESSED / "run_sample_arena.jsonl")}
    recs = {r["input_id"]: r for r in load_jsonl(PROCESSED / "run_results_arena_reprocessed.jsonl")}
    backfill = {d["input_id"]: d["strong_quality_score"] for d in load_jsonl(PROCESSED / "strong_baseline_backfill.jsonl")}

    pairs = sorted({k.rsplit("_", 1)[0] for k in recs})
    rows = []
    for pid in pairs:
        ia, ib = f"{pid}_a", f"{pid}_b"
        if ia not in recs or ib not in recs:
            continue
        out = sample[ia]["metadata"]["outcome"]
        human = "tie" if out in ("tie", "tie_bothbad") else ("a" if out == "win" else "b")
        sc = {}
        for tag, rid in (("a", ia), ("b", ib)):
            r = recs[rid]
            weak = max(1, min(10, round(r["ml_arbitrator_output"]["predicted_quality_score"])))
            router = r["verdict"]["quality_score"]
            strong = router if r["verdict"]["adjudicated"] else backfill.get(rid)
            sc[tag] = (weak, router, strong, r["ml_arbitrator_output"]["arbitration_confidence"])
        la, lb = len(sample[ia]["output_text"]), len(sample[ib]["output_text"])
        rows.append({"pair_id": pid, "human": human,
                     "weak": pref(sc["a"][0], sc["b"][0]),
                     "router": pref(sc["a"][1], sc["b"][1]),
                     "strong": pref(sc["a"][2], sc["b"][2]) if None not in (sc["a"][2], sc["b"][2]) else None,
                     "gap_router": abs(sc["a"][1] - sc["b"][1]),
                     "conf": min(sc["a"][3], sc["b"][3]),
                     "len_a": la, "len_b": lb})
    P = pd.DataFrame(rows)
    print(f"[arena] {len(P)} pairs, {int((P.human != 'tie').sum())} with a non-tie human vote")

    # sanity against the stored per-pair table
    old = pd.read_csv(ANALYSIS / "zheng_agreement_per_pair.csv").set_index("pair_id")
    P = P.set_index("pair_id")
    assert (old.loc[P.index, "human_pref"] == P["human"]).all(), "human labels differ from stored table"
    assert (old.loc[P.index, "arbiter_pref"] == P["router"]).all(), "router prefs differ from stored table"
    P = P.reset_index()

    S2 = P[P.human != "tie"].reset_index(drop=True)

    out = []

    def add(system, metric, k, n, ref=None):
        lo, hi = wilson(k, n)
        out.append({"system": system, "metric": metric, "correct": k, "n": n,
                    "value_pct": 100 * k / n, "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi,
                    "zheng_reference_pct": ref})
        print(f"  {system:8s} {metric:20s} {100*k/n:5.1f}%  [{100*lo:.1f}, {100*hi:.1f}]  (n={n})")

    for sys_ in ["weak", "router", "strong"]:
        d = P[P[sys_].notna()]
        add(sys_, "S1 (all pairs)", int((d[sys_] == d.human).sum()), len(d), ZHENG["S1"])
        d2 = S2[S2[sys_].notna()]
        add(sys_, "S2 (non-tie human)", int((d2[sys_] == d2.human).sum()), len(d2), ZHENG["S2"])
        dec = d2[d2[sys_] != "tie"]
        add(sys_, "Decisive accuracy", int((dec[sys_] == dec.human).sum()), len(dec))
        add(sys_, "Own tie rate on S2", int((d2[sys_] == "tie").sum()), len(d2))
    T = pd.DataFrame(out)
    save_table(T, "arena_agreement_with_intervals.csv")

    # paired differences between systems on S2 (bootstrap over pairs + exact McNemar)
    rng = np.random.default_rng(SEED)
    drows = []
    for a_, b_ in [("router", "weak"), ("strong", "weak"), ("strong", "router")]:
        d = S2[S2[a_].notna() & S2[b_].notna()]
        ca, cb = (d[a_] == d.human).values.astype(float), (d[b_] == d.human).values.astype(float)
        diff, lo, hi = paired_bootstrap_diff(ca, cb)
        n10, n01 = int(((ca == 1) & (cb == 0)).sum()), int(((ca == 0) & (cb == 1)).sum())
        p = 1.0 if n10 + n01 == 0 else binomtest(n10, n10 + n01, 0.5).pvalue
        drows.append({"comparison": f"{a_} minus {b_} (S2 agreement)", "difference_pct_points": 100 * diff,
                      "ci_lo": 100 * lo, "ci_hi": 100 * hi, "only_first_right": n10, "only_second_right": n01,
                      "mcnemar_exact_p": p, "n": len(d)})
    # PGR at the deployed threshold, bootstrap over pairs
    d = S2[S2.strong.notna()].reset_index(drop=True)
    cw, cr, cs = [(d[x] == d.human).values.astype(float) for x in ("weak", "router", "strong")]
    pg, den_small = [], 0
    for idx in rng.integers(0, len(d), (4000, len(d))):
        den = cs[idx].mean() - cw[idx].mean()
        if abs(den) < 0.02:
            den_small += 1
            continue
        pg.append(100 * (cr[idx].mean() - cw[idx].mean()) / den)
    pgr_point = 100 * (cr.mean() - cw.mean()) / (cs.mean() - cw.mean())
    drows.append({"comparison": "PGR at the deployed threshold (router vs weak/strong)",
                  "difference_pct_points": pgr_point,
                  "ci_lo": float(np.quantile(pg, 0.025)), "ci_hi": float(np.quantile(pg, 0.975)),
                  "only_first_right": np.nan, "only_second_right": np.nan, "mcnemar_exact_p": np.nan, "n": len(d)})
    print(f"  PGR point {pgr_point:.1f}, bootstrap 95% [{np.quantile(pg,0.025):.1f}, {np.quantile(pg,0.975):.1f}], "
          f"resamples dropped for near-zero denominator: {den_small}/4000")
    D = pd.DataFrame(drows)
    save_table(D, "arena_system_differences_with_intervals.csv")
    print(D.round(3).to_string())

    # agreement by score gap (router)
    bins = [("tie (gap 0)", lambda g: g == 0), ("gap 1", lambda g: g == 1), ("gap 2", lambda g: g == 2),
            ("gap 3-4", lambda g: (g >= 3) & (g <= 4)), ("gap 5 or more", lambda g: g >= 5)]
    grows = []
    for name, f in bins:
        d = S2[f(S2.gap_router)]
        k = int((d.router == d.human).sum())
        lo, hi = wilson(k, len(d))
        grows.append({"score_gap": name, "n_pairs": len(d), "agreement_pct": 100 * k / len(d),
                      "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi})
    G = pd.DataFrame(grows)
    save_table(G, "arena_agreement_by_score_gap_with_intervals.csv")
    print(G.round(1).to_string())

    # escalation confidence vs correctness
    corr = (S2.router == S2.human).astype(float).values
    r, p = pearsonr(S2.conf.values, corr)
    boot = [pearsonr(S2.conf.values[i], corr[i])[0] for i in rng.integers(0, len(S2), (2000, len(S2)))]
    C = pd.DataFrame([{"n": len(S2), "pearson_r_confidence_vs_correct": r, "p_value": p,
                       "ci_lo": float(np.nanquantile(boot, 0.025)), "ci_hi": float(np.nanquantile(boot, 0.975))}])
    save_table(C, "arena_escalation_confidence_vs_correctness.csv")
    print(C.round(3).to_string())

    # verbosity and slot checks
    S2["longer"] = np.where(S2.len_a > S2.len_b, "a", np.where(S2.len_b > S2.len_a, "b", "same"))
    V = S2[S2.longer != "same"]
    vrows = []
    k = int((V.human == V.longer).sum()); lo, hi = wilson(k, len(V))
    vrows.append({"who": "Human voters", "question": "picked the longer answer", "k": k, "n": len(V),
                  "pct": 100 * k / len(V), "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi})
    dec = V[V.router != "tie"]
    k = int((dec.router == dec.longer).sum()); lo, hi = wilson(k, len(dec))
    vrows.append({"who": "ARBITER (decisive pairs)", "question": "picked the longer answer", "k": k, "n": len(dec),
                  "pct": 100 * k / len(dec), "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi})
    k = int((dec.human == dec.longer).sum()); lo, hi = wilson(k, len(dec))
    vrows.append({"who": "Human voters (same decisive pairs)", "question": "picked the longer answer", "k": k, "n": len(dec),
                  "pct": 100 * k / len(dec), "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi})
    dd = P[(P.router != "tie")]
    k = int((dd.router == "a").sum()); lo, hi = wilson(k, len(dd))
    vrows.append({"who": "ARBITER (decisive pairs, all 130)", "question": "picked slot A", "k": k, "n": len(dd),
                  "pct": 100 * k / len(dd), "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi})
    k = int((S2.human == "a").sum()); lo, hi = wilson(k, len(S2))
    vrows.append({"who": "Human voters (non-tie)", "question": "picked slot A", "k": k, "n": len(S2),
                  "pct": 100 * k / len(S2), "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi})
    rho = spearmanr(np.log(S2.len_a / S2.len_b), (S2.human == "a").astype(int)).correlation
    Vt = pd.DataFrame(vrows)
    Vt["note"] = ""
    Vt.loc[0, "note"] = f"length ratio vs human vote Spearman {rho:.2f}"
    save_table(Vt, "arena_verbosity_and_slot_checks.csv")
    print(Vt.round(1).to_string())

    # ---- plots ----
    # 1) agreement with intervals and Zheng reference
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), sharex=True)
    for ax, metric, ttl in zip(axes, ["S1 (all pairs)", "S2 (non-tie human)", "Decisive accuracy"],
                               ["S1: all 130 pairs, ties count as a label",
                                "S2: the 85 pairs with a non-tie human vote",
                                "Decisive accuracy: S2 pairs where ARBITER did not tie"]):
        sub = T[T.metric == metric].set_index("system").loc[["weak", "router", "strong"]]
        yp = [2, 1, 0]
        for yy, (name, r) in zip(yp, sub.iterrows()):
            c = BLUE if name == "router" else GREY
            ax.plot([r.ci_lo_pct, r.ci_hi_pct], [yy, yy], color=c, lw=1.5)
            ax.plot(r.value_pct, yy, "o", color=c, ms=6)
            ax.text(r.value_pct, yy + 0.22, f"{r.value_pct:.1f}% (n={int(r.n)})", ha="center", va="bottom", fontsize=8.5)
        if metric in ("S1 (all pairs)", "S2 (non-tie human)"):
            ref = ZHENG["S1"] if metric.startswith("S1") else ZHENG["S2"]
            ax.axvline(ref, color=RED, ls="--", lw=1.2)
            ax.text(ref - 1, 3.45, f"Zheng et al.\nGPT-4 pairwise: {ref:.0f}%", ha="right", va="top", fontsize=8, color=RED)
        ax.axvline(50, color=LIGHTGREY, ls=":", lw=1)
        ax.set_yticks(yp)
        ax.set_yticklabels(["Weak (arbitrator only)", "ARBITER (router)", "Strong (adjudicator on all)"] if ax is axes[0] else [])
        ax.set_ylim(-0.5, 3.5)
        ax.set_xlim(20, 105)
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xlabel("Agreement with human vote (%)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Chatbot Arena: ARBITER agreement with human votes, 95% Wilson intervals (Zheng et al. numbers are from a different, larger run and protocol)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, "arena_agreement_with_human_votes_intervals_vs_zheng.png")
    plt.close(fig)

    # 2) by score gap
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(G))
    ax.bar(x, G.agreement_pct, 0.6, color=BLUE)
    for i, r in G.iterrows():
        ax.plot([i, i], [r.ci_lo_pct, r.ci_hi_pct], color="#222222", lw=1.1)
        ax.text(i, 3, f"n={int(r.n_pairs)}", ha="center", fontsize=8.5, color="white")
    ax.axhline(50, color=GREY, ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(G.score_gap)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Agreement with human vote (%)")
    ax.set_xlabel("Gap between the two answers' ARBITER scores")
    ax.set_title("Chatbot Arena: agreement rises with the score gap\n85 non-tie human pairs, 95% Wilson intervals", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    save_fig(fig, "arena_agreement_by_score_gap_with_intervals.png")
    plt.close(fig)

    # 3) verbosity
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    sub = Vt.iloc[:3]
    for i, (_, r) in enumerate(sub.iloc[::-1].iterrows()):
        c = BLUE if r.who.startswith("ARBITER") else GREY
        ax.plot([r.ci_lo_pct, r.ci_hi_pct], [i, i], color=c, lw=1.5)
        ax.plot(r.pct, i, "o", color=c, ms=6)
        ax.text(r.ci_hi_pct + 1.5, i, f"{r.pct:.0f}% (n={int(r.n)})", va="center", fontsize=8.5)
    ax.axvline(50, color=GREY, ls="--", lw=1)
    ax.set_yticks(range(3))
    ax.set_yticklabels(list(sub.who)[::-1])
    ax.set_xlim(20, 100)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("Share of pairs where the longer answer was preferred (%)")
    ax.set_title("Chatbot Arena: how often the longer answer wins (verbosity check)\n95% Wilson intervals; dashed line = no length preference", fontsize=10.5)
    ax.grid(axis="y", visible=False)
    save_fig(fig, "arena_verbosity_longer_answer_preferred_humans_vs_arbiter.png")
    plt.close(fig)


if __name__ == "__main__":
    main()