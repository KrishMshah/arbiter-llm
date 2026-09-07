"""
src/routing.py

Adaptive dimension router. Reads task_type, returns which DIMENSIONS apply.
All three critics always run (see critics.py) — this just scopes which
dimensions each one is asked to score, per task type.
"""

from src.schemas import TaskType, RoutingDecision

ALL_DIMENSIONS = ["factual_accuracy", "logical_consistency", "completeness"]

ROUTING_TABLE: dict[str, dict] = {
    TaskType.FACTUAL_QA.value: {
        "dimensions": ["factual_accuracy", "logical_consistency"],
        "skip": ["completeness"],
        "reason": "Factual QA does not require completeness evaluation",
    },
    TaskType.SUMMARISATION.value: {
        "dimensions": ["factual_accuracy", "logical_consistency", "completeness"],
        "skip": [],
        "reason": "All three dimensions are relevant for summarisation quality",
    },
    TaskType.REASONING.value: {
        "dimensions": ["logical_consistency", "completeness"],
        "skip": ["factual_accuracy"],
        "reason": "Reasoning tasks require logic and completeness; factual check less critical",
    },
    TaskType.CREATIVE.value: {
        "dimensions": ["logical_consistency", "completeness"],
        "skip": ["factual_accuracy"],
        "reason": "Creative tasks require logic and completeness; strict factual check not applicable",
    },
}


def route_critics(task_type: TaskType | str) -> RoutingDecision:
    key = task_type.value if isinstance(task_type, TaskType) else task_type

    if key not in ROUTING_TABLE:
        raise ValueError(f"Unknown task_type: {key!r}")

    rule = ROUTING_TABLE[key]
    return RoutingDecision(
        task_type=key,
        dimensions_selected=rule["dimensions"],
        dimensions_skipped=rule["skip"],
        routing_reason=rule["reason"],
    )