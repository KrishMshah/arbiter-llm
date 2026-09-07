"""
src/pipeline.py

LangGraph state machine wiring together routing, critics, disagreement
detection, ML arbitration, the adjudicator, and hallucination tracing.

Cost tracking lives here (not in critics.py/adjudicator.py) — centralizing
it keeps those files clean and means every dollar in the system is
accounted for in exactly one place. Token counts use tiktoken's o200k_base
encoding (GPT-4o's real tokenizer) as a universal approximator — exact for
Critic A and the adjudicator, an estimate for Claude Haiku and Llama since
no offline tokenizer exists for those without heavy extra dependencies.

Pricing verified via web search, current as of Aug 2026 — NOT the project
doc's numbers, which were stale for Claude Haiku (doc said ~$0.00025/1K
input, real rate is $0.0008/1K input + $0.004/1K output).

Cost estimation is skipped entirely in MOCK_MODE — no real API call was
made, so the true cost is $0, not "what it would have cost."

Phase 2 TODO (this file will need real work, not just a critics.py swap):
  - Critic failure handling: PARTIALLY done. dispatch_critics_node now
    catches any exception from critic.evaluate() and marks that critic
    failed (see critics.py's make_failed_critique + schemas.py's
    CritiqueOutput.critic_failed) instead of crashing — disagreement.py
    filters failed critics out of every comparison. STILL MISSING: the
    30-second timeout itself (nothing times out yet — mocks/no real calls
    exist to time out), and the "abort with error if 2+ critics fail"
    guardrail — right now the pipeline always continues regardless of how
    many critics failed, which is fine for 1 failure but wrong for 2+.
    Enforcing that abort needs ArbitrationState.error to distinguish
    "reject this request" from "degrade and continue" — it's currently one
    generic string, which isn't enough for that distinction.
  - Retry-once + simplified-fallback-prompt on critic schema validation
    failure — not applicable while critics are mocked
  - Cost guardrails: per-request $0.05 cap and daily $5 cap are computed
    here but NOT enforced/rejected yet — just tracked
  - Wrap run_pipeline() as POST /v1/arbitrate in FastAPI (Phase 4 per
    folder structure doc)
"""

import time
from typing import Optional, TypedDict

from src.config import MOCK_MODE
from src.schemas import (
    BenchmarkItem, RoutingDecision, CritiqueOutput, DisagreementMatrix,
    MLFeatures, MLArbitratorOutput, HallucinationTrace, Verdict, ArbitrationResult,
)
from src.routing import route_critics
from src.critics import CRITICS, make_failed_critique
from src.disagreement import detect_disagreement, extract_ml_features
from src.arbitrator import run_arbitrator
from src.adjudicator import run_adjudicator, retrieve_evidence
from src.hallucination import trace_hallucinations

from langgraph.graph import StateGraph, START, END


# ============================================================================
# Cost tracking — verified rates, Aug 2026. Llama/stub are $0 by design (local).
# ============================================================================

PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.00060},
    "claude-haiku-3.5": {"input_per_1k": 0.00080, "output_per_1k": 0.00400},
    "llama3.2:3b": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    "gpt-4o": {"input_per_1k": 0.00250, "output_per_1k": 0.01000},
    "stub": {"input_per_1k": 0.0, "output_per_1k": 0.0},
}

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("o200k_base")
except Exception:
    # Covers tiktoken not being installed AND get_encoding()'s first-use
    # network fetch failing (it downloads the BPE file from
    # openaipublic.blob.core.windows.net on first call, then caches it —
    # a network hiccup, firewall, or offline machine shouldn't break
    # importing this module. Falls back to the crude len(text)//4 estimate.
    _ENCODER = None


def count_tokens(text: str) -> int:
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return max(1, len(text) // 4)  # crude fallback if tiktoken isn't installed


def estimate_cost(model_used: str, input_text: str, output_text: str) -> float:
    if MOCK_MODE:
        return 0.0  # no real API call happened — nothing was actually billed
    rates = PRICING.get(model_used)
    if rates is None:
        return 0.0  # unknown model — don't silently guess a price
    input_cost = (count_tokens(input_text) / 1000) * rates["input_per_1k"]
    output_cost = (count_tokens(output_text) / 1000) * rates["output_per_1k"]
    return input_cost + output_cost


# ============================================================================
# Pipeline state
# ============================================================================

class ArbitrationState(TypedDict):
    input_id: str
    original_output: str
    task_type: str
    benchmark_item: Optional[BenchmarkItem]  # enables ground-truth-aware disagreement resolution

    routing_decision: Optional[RoutingDecision]

    critique_a: Optional[CritiqueOutput]
    critique_b: Optional[CritiqueOutput]
    critique_c: Optional[CritiqueOutput]

    disagreement_matrix: Optional[DisagreementMatrix]
    ml_features: Optional[MLFeatures]

    ml_arbitrator_output: Optional[MLArbitratorOutput]

    adjudicator_triggered: bool
    retrieved_evidence: Optional[list[str]]

    hallucination_trace: Optional[HallucinationTrace]

    verdict: Optional[Verdict]
    cost_breakdown: dict[str, float]
    total_cost_usd: float
    latency_ms: int
    start_time: float
    error: Optional[str]


# ============================================================================
# Nodes
# ============================================================================

def parse_input(state: ArbitrationState) -> dict:
    if not state.get("original_output", "").strip():
        return {"error": "original_output is empty"}
    return {"cost_breakdown": {}, "adjudicator_triggered": False, "error": None}


def route_critics_node(state: ArbitrationState) -> dict:
    decision = route_critics(state["task_type"])
    return {"routing_decision": decision}


def dispatch_critics_node(state: ArbitrationState) -> dict:
    dimensions = state["routing_decision"].dimensions_selected
    output_text = state["original_output"]
    cost_breakdown = dict(state.get("cost_breakdown", {}))

    critiques: dict[str, CritiqueOutput] = {}
    for critic_id, critic in CRITICS.items():
        try:
            critique = critic.evaluate(output_text, dimensions)
        except Exception as e:
            # One critic failing shouldn't crash the whole run — mark it and
            # continue. Whether 2+ failures should abort the pipeline entirely
            # is still open (Phase 2 TODO below) — this just makes sure a
            # single failure degrades gracefully instead of throwing.
            critique = make_failed_critique(critic_id, critic.model_used, str(e))
        critiques[critic_id] = critique
        cost_breakdown[critic_id] = estimate_cost(
            critic.model_used, input_text=output_text, output_text=critique.reasoning,
        )

    return {
        "critique_a": critiques["critic_a"],
        "critique_b": critiques["critic_b"],
        "critique_c": critiques["critic_c"],
        "cost_breakdown": cost_breakdown,
    }


def detect_disagreement_node(state: ArbitrationState) -> dict:
    critiques = [state["critique_a"], state["critique_b"], state["critique_c"]]
    matrix = detect_disagreement(critiques, benchmark_item=state.get("benchmark_item"))
    return {"disagreement_matrix": matrix}


def extract_ml_features_node(state: ArbitrationState) -> dict:
    critiques = [state["critique_a"], state["critique_b"], state["critique_c"]]
    features = extract_ml_features(critiques, state["disagreement_matrix"], state["task_type"])
    return {"ml_features": features}


def run_ml_arbitrator_node(state: ArbitrationState) -> dict:
    return {"ml_arbitrator_output": run_arbitrator(state["ml_features"])}


def should_escalate_to_gpt(state: ArbitrationState) -> str:
    return "escalate" if state["ml_arbitrator_output"].escalate_to_gpt else "accept"


def run_adjudicator_node(state: ArbitrationState) -> dict:
    critiques = [state["critique_a"], state["critique_b"], state["critique_c"]]
    evidence = retrieve_evidence(state["original_output"])
    verdict = run_adjudicator(
        state["original_output"], critiques, state["disagreement_matrix"], state["ml_arbitrator_output"],
    )

    cost_breakdown = dict(state.get("cost_breakdown", {}))
    cost_breakdown["adjudicator"] = estimate_cost(
        "gpt-4o",
        input_text=state["original_output"] + " ".join(evidence),
        output_text=verdict.adjudicator_reasoning or "",
    )

    return {
        "retrieved_evidence": evidence,
        "verdict": verdict,
        "adjudicator_triggered": True,
        "cost_breakdown": cost_breakdown,
    }


def trace_hallucinations_node(state: ArbitrationState) -> dict:
    critiques = [state["critique_a"], state["critique_b"], state["critique_c"]]
    trace = trace_hallucinations(state["original_output"], critiques)
    return {"hallucination_trace": trace}


def _ml_only_verdict(ml_output: MLArbitratorOutput, critiques: list[CritiqueOutput]) -> Verdict:
    """Used when the ML arbitrator's confidence was high enough that the
    adjudicator never ran (the fast path) — verdict has to come from
    somewhere, so it's built directly from the ML prediction."""
    all_issues = [i for c in critiques for i in c.issues]
    confidence_1_5 = max(1, min(5, round(1 + ml_output.arbitration_confidence * 4)))
    quality_1_10 = max(1, min(10, round(ml_output.predicted_quality_score)))

    return Verdict(
        quality_score=quality_1_10,
        confidence=confidence_1_5,
        confirmed_issues=all_issues,
        dismissed_flags=[],
        adjudicated=False,
        adjudicated_by="ml_arbitrator",
        adjudicator_reasoning=None,
    )


def assemble_verdict_node(state: ArbitrationState) -> dict:
    verdict = state.get("verdict")
    if verdict is None:  # fast path — adjudicator never ran
        critiques = [state["critique_a"], state["critique_b"], state["critique_c"]]
        verdict = _ml_only_verdict(state["ml_arbitrator_output"], critiques)

    cost_breakdown = state.get("cost_breakdown", {})
    total_cost = sum(cost_breakdown.values())
    latency_ms = int((time.time() - state["start_time"]) * 1000)

    return {
        "verdict": verdict,
        "total_cost_usd": round(total_cost, 6),
        "latency_ms": latency_ms,
    }


# ============================================================================
# Graph construction
# ============================================================================

def build_graph():
    graph = StateGraph(ArbitrationState)

    graph.add_node("parse_input", parse_input)
    graph.add_node("route_critics", route_critics_node)
    graph.add_node("dispatch_critics", dispatch_critics_node)
    graph.add_node("detect_disagreement", detect_disagreement_node)
    graph.add_node("extract_ml_features", extract_ml_features_node)
    graph.add_node("ml_arbitrator", run_ml_arbitrator_node)
    graph.add_node("adjudicator", run_adjudicator_node)
    graph.add_node("trace_hallucinations", trace_hallucinations_node)
    graph.add_node("assemble_verdict", assemble_verdict_node)

    graph.add_edge(START, "parse_input")
    graph.add_edge("parse_input", "route_critics")
    graph.add_edge("route_critics", "dispatch_critics")
    graph.add_edge("dispatch_critics", "detect_disagreement")
    graph.add_edge("detect_disagreement", "extract_ml_features")
    graph.add_edge("extract_ml_features", "ml_arbitrator")
    graph.add_conditional_edges(
        "ml_arbitrator", should_escalate_to_gpt,
        {"escalate": "adjudicator", "accept": "trace_hallucinations"},
    )
    graph.add_edge("adjudicator", "trace_hallucinations")
    graph.add_edge("trace_hallucinations", "assemble_verdict")
    graph.add_edge("assemble_verdict", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(
    input_id: str,
    original_output: str,
    task_type: str,
    benchmark_item: Optional[BenchmarkItem] = None,
) -> ArbitrationResult:
    """Direct call — no FastAPI yet (that's Phase 4). This IS the
    POST /v1/arbitrate behavior for now."""
    initial_state: ArbitrationState = {
        "input_id": input_id,
        "original_output": original_output,
        "task_type": task_type,
        "benchmark_item": benchmark_item,
        "routing_decision": None,
        "critique_a": None, "critique_b": None, "critique_c": None,
        "disagreement_matrix": None,
        "ml_features": None,
        "ml_arbitrator_output": None,
        "adjudicator_triggered": False,
        "retrieved_evidence": None,
        "hallucination_trace": None,
        "verdict": None,
        "cost_breakdown": {},
        "total_cost_usd": 0.0,
        "latency_ms": 0,
        "start_time": time.time(),
        "error": None,
    }

    final_state = get_graph().invoke(initial_state)

    if final_state.get("error"):
        raise ValueError(final_state["error"])

    return ArbitrationResult(
        input_id=final_state["input_id"],
        task_type=final_state["task_type"],
        original_output=final_state["original_output"],
        routing_decision=final_state["routing_decision"],
        critiques=[final_state["critique_a"], final_state["critique_b"], final_state["critique_c"]],
        disagreement_matrix=final_state["disagreement_matrix"],
        ml_features=final_state["ml_features"],
        ml_arbitrator_output=final_state["ml_arbitrator_output"],
        verdict=final_state["verdict"],
        hallucination_trace=final_state["hallucination_trace"],
        cost_breakdown=final_state["cost_breakdown"],
        total_cost_usd=final_state["total_cost_usd"],
        latency_ms=final_state["latency_ms"],
    )