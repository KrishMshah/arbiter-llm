"""
FinalScripts/tracer_origin_validation.py

Validates the deterministic hallucination tracer's ORIGIN sentences against the
FActScore annotation: each labelled biography comes with the list of sentences that
contain at least one unsupported fact (known_hallucination_spans).

Uses the tracer output already stored in run_results_combined_costed.jsonl (the tracer
is deterministic and has no LLM calls, so re-running it would give the same result).
Only origin sentences are validated. The gold labels mark unsupported sentences, not
causal chains, so propagation chains stay a stated heuristic.

Scoring (sentence level, pooled over the 174 items):
  precision = flagged origin sentences that are gold-unsupported / flagged origins
  recall    = gold-unsupported sentences that were flagged / gold-unsupported sentences
Baselines: flag every sentence; flag the same number of sentences at random (expected
value computed exactly); flag the first sentence only.
Intervals: bootstrap over items (sentences within an item are not independent).

Run: python FinalScripts/tracer_origin_validation.py
"""

import re
import numpy as np
import pandas as pd

from common import *

SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def match(a, b):
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def split_sentences(text):
    parts = []
    for para in text.split("\n"):
        parts += [s.strip() for s in SPLIT.split(para.strip()) if s.strip()]
    return parts


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * p * r / (p + r) if p and r and not np.isnan(p) and not np.isnan(r) else np.nan
    return p, r, f


def main():
    plt = setup_style()
    df, items, by_id = build_dataset()
    ids = df.loc[df["dataset"] == "factscore_labeled", "item_id"].tolist()

    recs = []
    for iid in ids:
        it, tr = items[iid], by_id[iid]["hallucination_trace"]
        gold = list(it["known_hallucination_spans"] or [])
        sents = split_sentences(it["output_text"])
        n_sent = max(len(sents), len(gold), 1)
        origins = tr["origin_sentences"]
        deps = tr["dependent_sentences"]
        o_hit = [any(match(o["text"], g) for g in gold) for o in origins]
        d_hit = [any(match(d["text"], g) for g in gold) for d in deps]
        g_hit = [any(match(o["text"], g) for o in origins) for g in gold]
        first_hit = int(len(gold) > 0 and len(sents) > 0 and any(match(sents[0], g) for g in gold))
        recs.append({"item_id": iid, "n_sentences": n_sent, "n_gold": len(gold),
                     "n_origins": len(origins), "tp_origin": int(sum(o_hit)),
                     "fp_origin": int(len(origins) - sum(o_hit)),
                     "fn_gold": int(len(gold) - sum(g_hit)),
                     "n_dependents": len(deps), "dependents_gold": int(sum(d_hit)),
                     "origin_flagger_counts": [len(o["flagged_by"]) for o in origins],
                     "origin_hits": o_hit, "first_sentence_gold": first_hit})
    R = pd.DataFrame(recs)
    N = len(R)
    print(f"[tracer] {N} FActScore items | gold-unsupported sentences {R.n_gold.sum()} of "
          f"{R.n_sentences.sum()} sentences ({100*R.n_gold.sum()/R.n_sentences.sum():.1f}%) | "
          f"origins flagged {R.n_origins.sum()} | items with >=1 origin {int((R.n_origins>0).sum())}")

    def metrics(idx):
        d = R.iloc[idx]
        tp, fp, fn = d.tp_origin.sum(), d.fp_origin.sum(), d.fn_gold.sum()
        p, r, f = prf(tp, fp, fn)
        # baselines on the same items
        G, S = d.n_gold.sum(), d.n_sentences.sum()
        all_p, all_r = G / S, 1.0
        all_f = 2 * all_p / (1 + all_p)
        # random flagging of the same number of sentences per item (exact expectation)
        e_tp = (d.n_origins * d.n_gold / d.n_sentences).sum()
        rp = e_tp / d.n_origins.sum()
        rr = e_tp / G
        rf = 2 * rp * rr / (rp + rr)
        # first sentence only
        ftp = d.first_sentence_gold.sum()
        fp1, fr1 = ftp / len(d), ftp / G
        ff1 = 2 * fp1 * fr1 / (fp1 + fr1) if fp1 + fr1 else np.nan
        return {"tracer": (p, r, f), "every sentence": (all_p, all_r, all_f),
                "random, same count": (rp, rr, rf), "first sentence only": (fp1, fr1, ff1)}

    point = metrics(np.arange(N))
    rng = np.random.default_rng(SEED)
    boots = [metrics(rng.integers(0, N, N)) for _ in range(1000)]
    rows = []
    for name in point:
        for k, mname in enumerate(["precision", "recall", "f1"]):
            vals = np.array([b[name][k] for b in boots], float)
            rows.append({"method": name, "metric": mname, "value": point[name][k],
                         "ci_lo": float(np.nanquantile(vals, 0.025)),
                         "ci_hi": float(np.nanquantile(vals, 0.975))})
    M = pd.DataFrame(rows)
    save_table(M, "tracer_origin_validation_vs_factscore_unsupported_sentences.csv")
    print(M.pivot(index="method", columns="metric", values="value").round(3))

    # tracer precision by how many critics flagged the origin sentence
    byk = {}
    for _, r in R.iterrows():
        for k, hit in zip(r["origin_flagger_counts"], r["origin_hits"]):
            a = byk.setdefault(k, [0, 0]); a[0] += int(hit); a[1] += 1
    krows = []
    for k in sorted(byk):
        h, n = byk[k]
        lo, hi = wilson(h, n)
        krows.append({"critics_flagging_the_sentence": k, "origin_sentences": n, "gold_unsupported": h,
                      "precision": h / n, "ci_lo": lo, "ci_hi": hi})
    K = pd.DataFrame(krows)
    save_table(K, "tracer_origin_precision_by_number_of_flagging_critics.csv")
    print(K.round(3).to_string())

    # dependents: how often is a "dependent" sentence itself an unsupported sentence?
    dep_n, dep_g = int(R.n_dependents.sum()), int(R.dependents_gold.sum())
    lo, hi = wilson(dep_g, dep_n) if dep_n else (np.nan, np.nan)
    base = R.n_gold.sum() / R.n_sentences.sum()
    D = pd.DataFrame([{"dependent_sentences": dep_n, "of_which_gold_unsupported": dep_g,
                       "share": dep_g / dep_n if dep_n else np.nan, "ci_lo": lo, "ci_hi": hi,
                       "base_rate_of_unsupported_sentences": base}])
    save_table(D, "tracer_dependent_sentences_vs_factscore_base_rate.csv")
    print(D.round(3).to_string())

    # item-level summary
    S = pd.DataFrame([{
        "items": N, "items_with_at_least_one_gold_sentence": int((R.n_gold > 0).sum()),
        "items_where_tracer_found_an_origin": int((R.n_origins > 0).sum()),
        "items_with_at_least_one_correct_origin": int((R.tp_origin > 0).sum()),
        "gold_sentences": int(R.n_gold.sum()), "all_sentences": int(R.n_sentences.sum()),
        "origin_sentences_flagged": int(R.n_origins.sum()),
        "origin_sentences_correct": int(R.tp_origin.sum())}])
    save_table(S, "tracer_origin_validation_counts.csv")
    print(S.T)

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    names = ["tracer", "first sentence only", "random, same count", "every sentence"]
    label = {"tracer": "Hallucination tracer (origins)", "first sentence only": "Always flag the first sentence",
             "random, same count": "Random sentences, same count", "every sentence": "Flag every sentence"}
    metr = ["precision", "recall", "f1"]
    w = 0.2
    for i, name in enumerate(names):
        for j, m in enumerate(metr):
            r = M[(M.method == name) & (M.metric == m)].iloc[0]
            c = BLUE if name == "tracer" else [GREY, LIGHTGREY, "#9a9a9a"][i - 1]
            x = j + (i - 1.5) * w
            ax.bar(x, r["value"], w * 0.92, color=c, label=label[name] if j == 0 else None)
            ax.plot([x, x], [r["ci_lo"], r["ci_hi"]], color="#222222", lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Share of sentences")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5, loc="upper left", ncol=2)
    ax.set_title(f"Origin sentences vs FActScore unsupported sentences\n{N} biographies, {int(R.n_sentences.sum())} sentences, "
                 f"{int(R.n_gold.sum())} unsupported; 95% intervals over biographies", fontsize=10.5)
    ax = axes[1]
    ax.bar(K["critics_flagging_the_sentence"].astype(str), K["precision"], 0.55, color=BLUE)
    for i, r in K.iterrows():
        ax.plot([i, i], [r["ci_lo"], r["ci_hi"]], color="#222222", lw=1)
        ax.text(i, 0.03, f"n={int(r['origin_sentences'])}", ha="center", fontsize=8.5, color="white")
    ax.axhline(base, color=RED, ls="--", lw=1.2, label="share of all sentences that are unsupported (49%)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Number of critics that flagged the origin sentence")
    ax.set_ylabel("Precision of the origin sentence")
    ax.grid(axis="x", visible=False)
    ax.set_title("Tracer precision by critic agreement", fontsize=10.5)
    fig.tight_layout()
    save_fig(fig, "tracer_origin_precision_recall_vs_factscore_unsupported_sentences.png")
    plt.close(fig)


if __name__ == "__main__":
    main()