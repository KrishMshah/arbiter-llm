"""
FinalScripts/pointwise_terra_adjudicator_759.py      (to-do item 10, PAID)

Runs the real gpt-5.6-terra adjudicator (src.adjudicator.run_adjudicator, unchanged)
on EVERY one of the 759 human-labelled items, reusing the stored critic outputs
(no critic is called again, so no Anthropic/Haiku cost). This gives:

  strong = terra's score on every item
  weak   = the arbitrator's score, out-of-fold (each item is predicted by an XGBoost
           model that never saw it: 5-fold stratified CV, 10 repeats, averaged;
           the same predictions the earlier ablation used)
  router = terra's score where the escalation rule fires, the arbitrator's otherwise
           (escalate when the hand-written confidence < ML_CONFIDENCE_THRESHOLD)

The adjudicator prompt shows terra the arbitrator's estimate, so that estimate is
the OUT-OF-FOLD one; no item is judged with a prediction from a model trained on it.
Same prompt as the arena run, so these numbers are comparable with it.

Items are processed in a fixed shuffled order, so a run that stops early still covers
all four datasets evenly. One JSON line per item, written immediately; re-running resumes.

Run from the repo root:
    python FinalScripts/pointwise_terra_adjudicator_759.py --estimate
    python FinalScripts/pointwise_terra_adjudicator_759.py --limit 3     # smoke test
    python FinalScripts/pointwise_terra_adjudicator_759.py               # full run (resumes)
    python FinalScripts/pointwise_terra_adjudicator_759.py --analyze     # summary only, no calls
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from common import *
from feature_group_ablation import cv_predictions, xgb_cond

HARD_CAP_USD = 6.00
CALL_MARGIN_USD = 0.03
CONSECUTIVE_FAILURE_LIMIT = 3
OUT_PATH = PROCESSED / "pointwise_terra_759.jsonl"
LEDGER_PATHS = [PROCESSED / "pairwise_terra_arena_swap.jsonl", OUT_PATH]   # shared budget

BUDGET_SIGNATURES = ["insufficient_quota", "insufficient quota", "billing_hard_limit",
                     "exceeded your current quota", "invalid_api_key", "credit balance"]


def spent_so_far():
    total = 0.0
    for p in LEDGER_PATHS:
        if p.exists():
            total += sum(r.get("cost_usd", 0.0) for r in load_jsonl(p))
    return total


def cost_of(in_tok, out_tok):
    from src.pipeline import cost_from_tokens
    return cost_from_tokens("gpt-5.6-terra", in_tok, out_tok, is_mock=False)


def build_inputs():
    """One dict per labelled item with the stored critic outputs and the out-of-fold
    arbitrator estimate, in a fixed shuffled order."""
    from src.arbitrator import compute_arbitration_confidence
    from src.config import ML_CONFIDENCE_THRESHOLD
    from src.schemas import CritiqueOutput, DisagreementMatrix, MLFeatures, MLArbitratorOutput

    df, items, by_id = build_dataset()
    print(f"[pointwise] {len(df)} labelled items; computing out-of-fold arbitrator predictions...")
    oof = np.clip(cv_predictions(df, xgb_cond(FEATURE_NAMES)).mean(axis=0), 1, 10)

    jobs = []
    for i, row in df.reset_index(drop=True).iterrows():
        rec = by_id[row["item_id"]]
        feats = MLFeatures(**rec["ml_features"])
        conf = compute_arbitration_confidence(feats)
        ml = MLArbitratorOutput(predicted_quality_score=round(float(oof[i]), 2), arbitration_confidence=conf,
                                model_used="xgboost", escalate_to_gpt=conf < ML_CONFIDENCE_THRESHOLD)
        jobs.append({"item_id": row["item_id"], "dataset": row["dataset"], "task_type": row["task_type"],
                     "human": float(row["y"]), "oof_pred": float(oof[i]), "confidence": conf,
                     "would_escalate": bool(ml.escalate_to_gpt),
                     "text": rec["original_output"],
                     "critiques": [CritiqueOutput(**c) for c in rec["critiques"]],
                     "matrix": DisagreementMatrix(**rec["disagreement_matrix"]), "ml": ml})
    rng = np.random.default_rng(SEED)
    return [jobs[i] for i in rng.permutation(len(jobs))]


def estimate(jobs):
    from src.adjudicator import _build_prompt
    toks = []
    for j in jobs:
        idx = [(k, c.critic_id, issue) for k, (c, issue) in enumerate(
            (c, issue) for c in j["critiques"] for issue in c.issues)]
        toks.append(len(_build_prompt(j["text"], idx, j["matrix"], j["ml"])) / 4)
    cost = sum(t / 1000 * 0.002 + 200 / 1000 * 0.012 for t in toks)
    print(f"[estimate] {len(jobs)} calls, about {np.mean(toks):.0f} input tokens each (chars/4), 200 output tokens "
          f"assumed -> about ${cost:.2f} (the arena run averaged $0.0043 per call). "
          f"Spent so far (items 9+10): ${spent_so_far():.3f}")


def run(jobs, limit, cap):
    from src.adjudicator import run_adjudicator
    done = {r["item_id"] for r in load_jsonl(OUT_PATH)} if OUT_PATH.exists() else set()
    todo = [j for j in jobs if j["item_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[pointwise] {len(done)}/{len(jobs)} already saved, {len(todo)} to do. "
          f"Spent so far: ${spent_so_far():.3f}, cap ${cap:.2f}")
    if not todo:
        return
    fails, n_ok, session_cost = 0, 0, 0.0
    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for i, j in enumerate(todo, 1):
            if spent_so_far() + CALL_MARGIN_USD >= cap:
                print(f"[pointwise] STOPPED: spend would reach the ${cap:.2f} cap. Re-run resumes.")
                break
            try:
                verdict, tin, tout = run_adjudicator(j["text"], j["critiques"], j["matrix"], j["ml"])
            except Exception as e:
                fails += 1
                msg = str(e)
                hint = "  <-- looks like a billing/quota/key problem" if any(s in msg.lower() for s in BUDGET_SIGNATURES) else ""
                print(f"[pointwise] [{i}/{len(todo)}] FAILED {j['item_id']}: {msg[:300]}{hint}")
                if fails >= CONSECUTIVE_FAILURE_LIMIT:
                    print(f"[pointwise] STOPPED: {fails} failures in a row. Fix and re-run; nothing is repeated.")
                    break
                time.sleep(2)
                continue
            fails = 0
            c = cost_of(tin, tout)
            session_cost += c
            out.write(json.dumps({"item_id": j["item_id"], "dataset": j["dataset"], "task_type": j["task_type"],
                                  "human": j["human"], "oof_pred": j["oof_pred"], "confidence": j["confidence"],
                                  "would_escalate": j["would_escalate"], "terra_score": verdict.quality_score,
                                  "terra_confidence": verdict.confidence,
                                  "n_confirmed": len(verdict.confirmed_issues), "n_dismissed": len(verdict.dismissed_flags),
                                  "input_tokens": tin, "output_tokens": tout, "cost_usd": round(c, 6)}) + "\n")
            out.flush()
            n_ok += 1
            if i % 25 == 0 or i == len(todo):
                print(f"[pointwise] [{i}/{len(todo)}] this session ${session_cost:.3f}, total spent ${spent_so_far():.3f}")
    print(f"[pointwise] session done: {n_ok} items, ${session_cost:.3f}")


def summarise():
    """Short summary only. PGR / APGR / CPT and the cost-quality curve are item 11."""
    if not OUT_PATH.exists():
        print("[summary] no results file yet")
        return
    R = pd.DataFrame(load_jsonl(OUT_PATH))
    R["router"] = np.where(R.would_escalate, R.terra_score, np.clip(np.round(R.oof_pred), 1, 10))
    R["weak"] = np.clip(np.round(R.oof_pred), 1, 10)
    R["strong"] = R.terra_score
    print(f"[summary] {len(R)} items judged by terra; escalation rule fires on {100*R.would_escalate.mean():.1f}%; "
          f"cost ${R.cost_usd.sum():.3f} (${R.cost_usd.mean():.4f} per item)")
    rows = []
    for name, sub in [("all items", R)] + [(DATASET_LABEL.get(d, d), g) for d, g in R.groupby("dataset")]:
        row = {"subset": name, "n": len(sub), "escalated_pct": 100 * sub.would_escalate.mean()}
        for s in ("weak", "router", "strong"):
            err = (sub[s] - sub.human).abs().values
            lo, hi = bootstrap_ci(err)
            row[f"{s}_mae"], row[f"{s}_lo"], row[f"{s}_hi"] = err.mean(), lo, hi
        rows.append(row)
    S = pd.DataFrame(rows)
    save_table(S, "pointwise_terra_759_error_vs_human_score_by_dataset.csv")
    print(S.round(3).to_string(index=False))
    print("(weak = out-of-fold arbitrator, strong = terra on every item, router = terra only where the "
          "escalation rule fires. Lower error is better. PGR/APGR/CPT come in item 11.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true", help="print the cost estimate and exit (no calls)")
    ap.add_argument("--analyze", action="store_true", help="summary only (no calls)")
    ap.add_argument("--limit", type=int, default=0, help="only the next N items (smoke test)")
    ap.add_argument("--cap", type=float, default=HARD_CAP_USD, help="shared hard cap in dollars")
    args = ap.parse_args()
    if args.analyze:
        summarise()
    else:
        jobs = build_inputs()
        if args.estimate:
            estimate(jobs)
        else:
            run(jobs, args.limit, min(args.cap, HARD_CAP_USD))
            summarise()