"""
FinalScripts/pairwise_terra_arena_swap_protocol.py 

Zheng-style pairwise judging of the 130 Chatbot Arena pairs with gpt-5.6-terra:
the judge sees the user question and BOTH answers, once as (A=first, B=second)
and once with the two answers swapped. A pair counts as a win only if both orders
agree; otherwise it is a tie (Zheng et al., 2023).

Compared with the existing pointwise run (answers scored one at a time, question
not shown) the differences are bundled: question visibility, joint view, swap.
The paper says so; this script does not try to separate them.

What it does
  1. Calls terra 2 x 130 = 260 times (resumable, one JSON line per call).
  2. Prints and saves S1 / S2 / decisive agreement with human votes (Wilson 95%),
     order-A-B-only vs order-B-A-only vs swap-consistent, flip rate, first-position
     preference, verbosity check, and a paired comparison with the pointwise systems.
  3. Draws one plot.

Run from the repo root (the folder that contains src/ and results/):
    python FinalScripts/pairwise_terra_arena_swap_protocol.py --estimate
    python FinalScripts/pairwise_terra_arena_swap_protocol.py --limit 2     # smoke test, 4 calls
    python FinalScripts/pairwise_terra_arena_swap_protocol.py               # full run (resumes)
    python FinalScripts/pairwise_terra_arena_swap_protocol.py --analyze     # analysis only, no calls
"""

import argparse
import json
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import *

MODEL = "gpt-5.6-terra"
HARD_CAP_USD = 6.00
CALL_MARGIN_USD = 0.03          # stop if spent + margin would reach the cap
MAX_CHARS = 6000                # same truncation as src/adjudicator.py
CONSECUTIVE_FAILURE_LIMIT = 3

OUT_PATH = PROCESSED / "pairwise_terra_arena_swap.jsonl"
LEDGER_PATHS = [OUT_PATH, PROCESSED / "pointwise_terra_759.jsonl"]   # shared budget

BUDGET_SIGNATURES = ["insufficient_quota", "insufficient quota", "billing_hard_limit",
                     "exceeded your current quota", "invalid_api_key", "credit balance"]

# Judge prompt: the pairwise instruction from Zheng et al. (2023), Fig. 5, with one
# addition (explanation kept short) to bound the output cost.
INSTRUCTION = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI "
    "assistants to the user question displayed below. You should choose the assistant that follows the "
    "user's instructions and answers the user's question better. Your evaluation should consider factors "
    "such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their "
    "responses. Begin your evaluation by comparing the two responses and provide a short explanation "
    "(at most 80 words). Avoid any position biases and ensure that the order in which the responses were "
    "presented does not influence your decision. Do not allow the length of the responses to influence "
    "your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After "
    "providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" "
    "if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."
)


# ------------------------------------------------------------------ data
def load_pairs():
    items = load_jsonl(PROCESSED / "run_sample_arena.jsonl")
    pairs = {}
    for it in items:
        pid, side = it["item_id"].rsplit("_", 1)
        pairs.setdefault(pid, {})[side] = it
    return {p: v for p, v in sorted(pairs.items()) if "a" in v and "b" in v}


def human_label(pair):
    out = pair["a"]["metadata"]["outcome"]
    return "tie" if out in ("tie", "tie_bothbad") else ("a" if out == "win" else "b")


def question_text(item):
    """The user turn(s) the answers respond to. Multi-turn pairs get the earlier turns too."""
    conv = item["metadata"].get("full_conversation") or []
    if len(conv) > 1 and conv[-1].get("role") == "assistant":
        ctx = conv[:-1]
        if len(ctx) == 1:
            return ctx[0]["content"][:MAX_CHARS]
        return "\n\n".join(f'{"User" if m["role"] == "user" else "Assistant"}: {m["content"][:MAX_CHARS]}'
                           for m in ctx)
    return item["input_prompt"][:MAX_CHARS]


def build_prompt(pair, order):
    first, second = (pair["a"], pair["b"]) if order == "ab" else (pair["b"], pair["a"])
    return (f"[System]\n{INSTRUCTION}\n\n[User Question]\n{question_text(pair['a'])}\n\n"
            f"[The Start of Assistant A's Answer]\n{first['output_text'][:MAX_CHARS]}\n[The End of Assistant A's Answer]\n\n"
            f"[The Start of Assistant B's Answer]\n{second['output_text'][:MAX_CHARS]}\n[The End of Assistant B's Answer]")


def parse_verdict(text):
    m = re.findall(r"\[\[\s*([ABC])\s*\]\]", text or "")
    return m[-1] if m else None


# ------------------------------------------------------------------ money
def spent_so_far():
    total = 0.0
    for p in LEDGER_PATHS:
        if p.exists():
            total += sum(r.get("cost_usd", 0.0) for r in load_jsonl(p))
    return total


_client = None


def call_terra(prompt):
    """Returns (text, input_tokens, output_tokens). Replaced by a fake in offline tests."""
    global _client
    if _client is None:
        from openai import OpenAI
        from src.config import OPENAI_API_KEY
        _client = OpenAI(api_key=OPENAI_API_KEY)
    resp = _client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        reasoning_effort="none", max_completion_tokens=400)
    return (resp.choices[0].message.content or ""), resp.usage.prompt_tokens, resp.usage.completion_tokens


def cost_of(in_tok, out_tok):
    from src.pipeline import cost_from_tokens
    return cost_from_tokens(MODEL, in_tok, out_tok, is_mock=False)


# ------------------------------------------------------------------ run
def estimate(pairs):
    n_tok = [len(build_prompt(p, o)) / 4 for p in pairs.values() for o in ("ab", "ba")]
    cost = sum(t / 1000 * 0.002 + 130 / 1000 * 0.012 for t in n_tok)
    print(f"[estimate] {len(n_tok)} calls, about {np.mean(n_tok):.0f} input tokens each (chars/4), "
          f"130 output tokens assumed -> about ${cost:.2f}. Spent so far (items 9+10): ${spent_so_far():.3f}")


def run(pairs, limit, cap):
    done = {r["key"] for r in load_jsonl(OUT_PATH)} if OUT_PATH.exists() else set()
    pids = sorted(pairs)[:limit] if limit else sorted(pairs)
    jobs = [(pid, o) for pid in pids for o in ("ab", "ba") if f"{pid}|{o}" not in done]
    print(f"[pairwise] {len(done)} calls already saved, {len(jobs)} to do. Spent so far: ${spent_so_far():.3f}, cap ${cap:.2f}")
    if not jobs:
        return
    fails, n_ok, session_cost = 0, 0, 0.0
    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for i, (pid, order) in enumerate(jobs, 1):
            if spent_so_far() + CALL_MARGIN_USD >= cap:
                print(f"[pairwise] STOPPED: spend would reach the ${cap:.2f} cap. Re-run resumes.")
                break
            prompt = build_prompt(pairs[pid], order)
            try:
                text, tin, tout = call_terra(prompt)
            except Exception as e:
                fails += 1
                msg = str(e)
                hint = "  <-- looks like a billing/quota/key problem" if any(s in msg.lower() for s in BUDGET_SIGNATURES) else ""
                print(f"[pairwise] [{i}/{len(jobs)}] FAILED {pid} {order}: {msg[:300]}{hint}")
                if fails >= CONSECUTIVE_FAILURE_LIMIT:
                    print(f"[pairwise] STOPPED: {fails} failures in a row. Fix and re-run; nothing is repeated.")
                    break
                time.sleep(2)
                continue
            fails = 0
            v = parse_verdict(text)
            slot = {"A": "first", "B": "second", "C": "tie"}.get(v)
            if slot is None:
                winner = None
            elif slot == "tie":
                winner = "tie"
            else:
                winner = ("a" if order == "ab" else "b") if slot == "first" else ("b" if order == "ab" else "a")
            c = cost_of(tin, tout)
            session_cost += c
            out.write(json.dumps({"key": f"{pid}|{order}", "pair_id": pid, "order": order, "verdict": v,
                                  "slot_choice": slot, "winner": winner, "input_tokens": tin,
                                  "output_tokens": tout, "cost_usd": round(c, 6), "reply": text}) + "\n")
            out.flush()
            n_ok += 1
            if i % 10 == 0 or i == len(jobs):
                print(f"[pairwise] [{i}/{len(jobs)}] this session ${session_cost:.3f}, total spent ${spent_so_far():.3f}")
    print(f"[pairwise] session done: {n_ok} calls, ${session_cost:.3f}")


# ------------------------------------------------------------------ analysis
def mcnemar(x, y):
    """x, y: boolean arrays (correct under system 1 / system 2). Exact two-sided test."""
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)


def analyse(pairs):
    if not OUT_PATH.exists():
        print("[analysis] no results file yet")
        return
    R = pd.DataFrame(load_jsonl(OUT_PATH))
    R["winner"] = R["winner"].fillna("tie")        # unparseable verdicts count as a tie, and are reported
    n_bad = int(R["verdict"].isna().sum())
    wide = R.pivot(index="pair_id", columns="order", values="winner").dropna()
    pids = list(wide.index)
    print(f"[analysis] {len(pids)} pairs with both orders ({len(R)} calls, {n_bad} unparseable verdicts)")
    if len(pids) < 20:
        print("[analysis] fewer than 20 complete pairs -- skipping the tables and plot (smoke test).")
        return

    # pointwise comparators from the stored run
    recs = {r["input_id"]: r for r in load_jsonl(PROCESSED / "run_results_arena_reprocessed.jsonl")}
    backfill = {d["input_id"]: d["strong_quality_score"] for d in load_jsonl(PROCESSED / "strong_baseline_backfill.jsonl")}

    def pref(a, b):
        return "a" if a > b else ("b" if b > a else "tie")

    rows = []
    for pid in pids:
        pr = pairs[pid]
        ia, ib = f"{pid}_a", f"{pid}_b"
        router = pref(recs[ia]["verdict"]["quality_score"], recs[ib]["verdict"]["quality_score"])
        sa = recs[ia]["verdict"]["quality_score"] if recs[ia]["verdict"]["adjudicated"] else backfill[ia]
        sb = recs[ib]["verdict"]["quality_score"] if recs[ib]["verdict"]["adjudicated"] else backfill[ib]
        w_ab, w_ba = wide.loc[pid, "ab"], wide.loc[pid, "ba"]
        rows.append({"pair_id": pid, "human": human_label(pr), "router_pointwise": router,
                     "terra_pointwise": pref(sa, sb), "terra_pair_AB": w_ab, "terra_pair_BA": w_ba,
                     "terra_pair_swap": w_ab if w_ab == w_ba else "tie",
                     "len_a": len(pr["a"]["output_text"]), "len_b": len(pr["b"]["output_text"])})
    P = pd.DataFrame(rows)
    S2 = P[P.human != "tie"].reset_index(drop=True)

    systems = [("router_pointwise", "ARBITER router, pointwise (no question shown)"),
               ("terra_pointwise", "terra alone, pointwise (no question shown)"),
               ("terra_pair_AB", "terra pairwise, one order only (A-B)"),
               ("terra_pair_BA", "terra pairwise, one order only (B-A)"),
               ("terra_pair_swap", "terra pairwise, both orders must agree (Zheng)")]
    out = []

    def add(label, metric, k, n):
        lo, hi = wilson(k, n)
        out.append({"system": label, "metric": metric, "correct": int(k), "n": int(n),
                    "pct": 100 * k / n if n else np.nan, "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi})

    for col, label in systems:
        add(label, "S1 (all pairs, a human tie is a label)", (P[col] == P.human).sum(), len(P))
        add(label, "S2 (non-tie human pairs; a system tie is wrong)", (S2[col] == S2.human).sum(), len(S2))
        dec = S2[S2[col] != "tie"]
        add(label, "decisive accuracy (S2 pairs where the system did not tie)", (dec[col] == dec.human).sum(), len(dec))
        add(label, "tie rate on S2 pairs", (S2[col] == "tie").sum(), len(S2))
    T = pd.DataFrame(out)
    save_table(T, "arena_pairwise_swap_terra_agreement_with_human_votes.csv")
    print(T.round(1).to_string(index=False))

    # position behaviour, from the raw calls (all pairs with both orders)
    Rp = R[R.pair_id.isin(pids)]
    nontie = Rp[Rp.slot_choice.isin(["first", "second"])]
    k_first = int((nontie.slot_choice == "first").sum())
    lo, hi = wilson(k_first, len(nontie))
    inconsistent = int((P.terra_pair_AB != P.terra_pair_BA).sum())
    slot_locked = 0
    for pid in pids:
        a = Rp[(Rp.pair_id == pid) & (Rp.order == "ab")].iloc[0].slot_choice
        b = Rp[(Rp.pair_id == pid) & (Rp.order == "ba")].iloc[0].slot_choice
        slot_locked += int(a == b and a in ("first", "second"))
    li, hi_i = wilson(inconsistent, len(P))
    ls, hs = wilson(slot_locked, len(P))
    pos = pd.DataFrame([
        {"measure": "pairs where the two orders disagree (flip rate)", "k": inconsistent, "n": len(P),
         "pct": 100 * inconsistent / len(P), "ci_lo_pct": 100 * li, "ci_hi_pct": 100 * hi_i},
        {"measure": "pairs where the judge picked the same SLOT in both orders (pure position following)", "k": slot_locked,
         "n": len(P), "pct": 100 * slot_locked / len(P), "ci_lo_pct": 100 * ls, "ci_hi_pct": 100 * hs},
        {"measure": "first slot chosen, among non-tie calls (50% = no first-position preference)", "k": k_first,
         "n": len(nontie), "pct": 100 * k_first / len(nontie), "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi},
    ])
    save_table(pos, "arena_pairwise_swap_terra_position_behaviour.csv")
    print(pos.round(1).to_string(index=False))

    # paired comparison on S2 correctness
    comp = []
    swap_ok = (S2.terra_pair_swap == S2.human).values
    for col, label in [("router_pointwise", "ARBITER router, pointwise"), ("terra_pointwise", "terra alone, pointwise"),
                       ("terra_pair_AB", "terra pairwise, order A-B only")]:
        other = (S2[col] == S2.human).values
        d, dlo, dhi = paired_bootstrap_diff(swap_ok.astype(float), other.astype(float))
        b, c, p = mcnemar(swap_ok, other)
        comp.append({"swap_consistent_pairwise_minus": label, "difference_pct_points": 100 * d,
                     "ci_lo": 100 * dlo, "ci_hi": 100 * dhi, "pairs_pairwise_right_other_wrong": b,
                     "pairs_other_right_pairwise_wrong": c, "mcnemar_exact_p": p, "n": len(S2)})
    C = pd.DataFrame(comp)
    save_table(C, "arena_pairwise_swap_terra_paired_differences_on_s2.csv")
    print(C.round(3).to_string(index=False))

    # verbosity: does the longer answer win?
    V = []
    d = S2[S2.len_a != S2.len_b].copy()
    d["longer"] = np.where(d.len_a > d.len_b, "a", "b")
    for label, col in [("human votes", "human"), ("terra pairwise (swap)", "terra_pair_swap"),
                       ("terra pointwise", "terra_pointwise"), ("ARBITER router", "router_pointwise")]:
        dd = d[d[col] != "tie"]
        k = int((dd[col] == dd.longer).sum())
        lo, hi = wilson(k, len(dd))
        V.append({"who": label, "longer_answer_preferred": k, "n_decisive": len(dd), "pct": 100 * k / len(dd),
                  "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi})
    V = pd.DataFrame(V)
    save_table(V, "arena_pairwise_swap_terra_verbosity_check.csv")
    print(V.round(1).to_string(index=False))
    P.to_csv(TABLES / "arena_pairwise_swap_terra_per_pair.csv", index=False)

    # ---- plot ----
    plt = setup_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    pick = [(systems[0][1], BLUE), (systems[1][1], "#7aa6d6"), (systems[2][1], LIGHTGREY), (systems[4][1], ORANGE)]
    metrics = ["S1 (all pairs, a human tie is a label)", "S2 (non-tie human pairs; a system tie is wrong)",
               "decisive accuracy (S2 pairs where the system did not tie)"]
    short = ["S1: all pairs\n(human tie is a label)", "S2: non-tie human pairs\n(system tie counts wrong)",
             "Decisive accuracy\n(system did not tie)"]
    w = 0.2
    for j, (label, colr) in enumerate(pick):
        for i, m in enumerate(metrics):
            r = T[(T.system == label) & (T.metric == m)].iloc[0]
            x = i + (j - 1.5) * w
            ax.bar(x, r.pct, w * 0.9, color=colr, label=label if i == 0 else None)
            ax.plot([x, x], [r.ci_lo_pct, r.ci_hi_pct], color="#222222", lw=1)
            ax.text(x, r.ci_hi_pct + 1.5, f"{r.pct:.0f}", ha="center", fontsize=8)
    for i, ref in enumerate([64, 87]):
        ax.plot([i - 0.42, i + 0.42], [ref, ref], color=RED, lw=1.6, ls="--",
                label="Zheng et al. GPT-4 pairwise, their Arena data" if i == 0 else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels(short)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Agreement with human votes (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax.set_title(f"Chatbot Arena: pointwise scoring vs pairwise judging with position swap (gpt-5.6-terra)\n"
                 f"{len(P)} pairs, {len(S2)} with a non-tie human vote; bars show 95% Wilson intervals", fontsize=10.5)
    fig.tight_layout()
    save_fig(fig, "arena_pairwise_swap_vs_pointwise_agreement_with_human_votes.png")
    plt.close(fig)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true", help="print the cost estimate and exit (no calls)")
    ap.add_argument("--analyze", action="store_true", help="analysis only (no calls)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pairs (smoke test)")
    ap.add_argument("--cap", type=float, default=HARD_CAP_USD, help="shared hard cap in dollars")
    args = ap.parse_args()
    pairs = load_pairs()
    print(f"[pairwise] {len(pairs)} complete arena pairs")
    if args.estimate:
        estimate(pairs)
    elif args.analyze:
        analyse(pairs)
    else:
        run(pairs, args.limit, min(args.cap, HARD_CAP_USD))
        analyse(pairs)