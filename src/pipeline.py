"""
src/pipeline.py

LangGraph state machine: routing -> critics -> disagreement -> ML
arbitration -> adjudicator (if escalated) -> hallucination trace -> verdict.
"""

import time
from typing import Optional, TypedDict

from src.config import MOCK_CRITIC_A, MOCK_CRITIC_B, MOCK_CRITIC_C
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

from langgraph.graph import StateGraph, START, END # type: ignore


PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.00060},
    "claude-haiku-4-5": {"input_per_1k": 0.00100, "output_per_1k": 0.00500},
    "llama3.2:3b": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    "gpt-5.6-terra": {"input_per_1k": 0.00200, "output_per_1k": 0.01200},
    "stub": {"input_per_1k": 0.0, "output_per_1k": 0.0},
}

CRITIC_MOCK_FLAGS = {
    "critic_a": MOCK_CRITIC_A,
    "critic_b": MOCK_CRITIC_B,
    "critic_c": MOCK_CRITIC_C,
}

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("o200k_base")
except Exception:
    _ENCODER = None  # offline/no network on first use — fall back to len//4


def count_tokens(text: str) -> int:
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return max(1, len(text) // 4)


def estimate_cost(model_used: str, input_text: str, output_text: str, is_mock: bool = False) -> float:
    if is_mock:
        return 0.0
    rates = PRICING.get(model_used)
    if rates is None:
        return 0.0
    input_cost = (count_tokens(input_text) / 1000) * rates["input_per_1k"]
    output_cost = (count_tokens(output_text) / 1000) * rates["output_per_1k"]
    return input_cost + output_cost


class ArbitrationState(TypedDict):
    input_id: str
    original_output: str
    task_type: str
    benchmark_item: Optional[BenchmarkItem]

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
            critique = make_failed_critique(critic_id, critic.model_used, str(e))
        critiques[critic_id] = critique
        cost_breakdown[critic_id] = estimate_cost(
            critic.model_used, input_text=output_text, output_text=critique.reasoning,
            is_mock=CRITIC_MOCK_FLAGS.get(critic_id, False),
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
        "gpt-5.6-terra",
        input_text=state["original_output"] + " ".join(evidence),
        output_text=verdict.adjudicator_reasoning or "",
        is_mock=True,  # run_adjudicator() is still a Phase 1 stub
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