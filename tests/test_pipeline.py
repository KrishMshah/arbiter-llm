"""
tests/test_pipeline.py

Integration test on 20 fixture outputs, per Phase 1 spec. Runs each fixture
through the full pipeline and checks the result is a valid, complete
ArbitrationResult — the actual Phase 1 "done" condition.
"""

import json
from pathlib import Path

import pytest

from unittest.mock import patch

from src.config import MOCK_MODE
from src.pipeline import run_pipeline
from src.routing import ROUTING_TABLE
from src.schemas import ArbitrationResult, BenchmarkItem, DatasetSource, TaskType, QualityScoreBasis
from src.critics import CRITICS

FIXTURES_PATH = Path(__file__).parent / "fixtures.json"


def load_fixtures() -> list[dict]:
    with open(FIXTURES_PATH) as f:
        return json.load(f)


FIXTURES = load_fixtures()


def test_fixtures_file_has_20_items():
    assert len(FIXTURES) == 20


@pytest.mark.parametrize("fixture", FIXTURES, ids=[f["id"] for f in FIXTURES])
def test_pipeline_runs_end_to_end(fixture):
    result = run_pipeline(
        input_id=fixture["id"],
        original_output=fixture["output_text"],
        task_type=fixture["task_type"],
    )

    assert isinstance(result, ArbitrationResult)

    # Verdict is well-formed
    assert 1 <= result.verdict.quality_score <= 10
    assert 1 <= result.verdict.confidence <= 5
    assert result.verdict.adjudicated_by in ("ml_arbitrator", "gpt4o_adjudicator")

    # All 3 critics always run, scoped to the routed dimensions
    assert len(result.critiques) == 3
    expected_dims = set(ROUTING_TABLE[fixture["task_type"]]["dimensions"])
    for critique in result.critiques:
        assert set(critique.dimensions_evaluated) == expected_dims
        assert set(critique.dimension_scores.keys()) == expected_dims

    # Routing decision matches the routing table for this task type
    assert result.routing_decision.task_type == fixture["task_type"]
    assert set(result.routing_decision.dimensions_selected) == expected_dims

    # Hallucination trace always present, never crashes
    assert result.hallucination_trace is not None
    if result.hallucination_trace.has_hallucination:
        assert len(result.hallucination_trace.origin_sentences) > 0

    # Cost is zero in MOCK_MODE — no real API calls were made
    if MOCK_MODE:
        assert result.total_cost_usd == 0.0
        assert all(c == 0.0 for c in result.cost_breakdown.values())

    assert result.latency_ms >= 0


def test_both_arbitrator_paths_get_exercised():
    """With 20 fixtures and a 0.75 confidence threshold on deterministically
    seeded random confidence, at least one fixture should escalate to the
    adjudicator and at least one should take the ML fast path — confirming
    both branches of the conditional edge actually run somewhere in the suite."""
    results = [
        run_pipeline(input_id=f["id"], original_output=f["output_text"], task_type=f["task_type"])
        for f in FIXTURES
    ]
    adjudicated = [r for r in results if r.verdict.adjudicated_by == "gpt4o_adjudicator"]
    ml_only = [r for r in results if r.verdict.adjudicated_by == "ml_arbitrator"]

    assert len(adjudicated) > 0, "no fixture escalated — conditional edge's 'escalate' branch never ran"
    assert len(ml_only) > 0, "every fixture escalated — conditional edge's 'accept' branch never ran"


def test_empty_output_raises_error():
    with pytest.raises(ValueError, match="empty"):
        run_pipeline(input_id="empty_test", original_output="", task_type="factual_qa")


def test_same_input_is_deterministic():
    """Mocks are seeded off the input text — same input should always
    produce the same verdict, since nothing in Phase 1 is truly random."""
    fixture = FIXTURES[0]
    result_1 = run_pipeline(input_id=fixture["id"], original_output=fixture["output_text"], task_type=fixture["task_type"])
    result_2 = run_pipeline(input_id=fixture["id"], original_output=fixture["output_text"], task_type=fixture["task_type"])

    assert result_1.verdict.quality_score == result_2.verdict.quality_score
    assert result_1.ml_arbitrator_output.arbitration_confidence == result_2.ml_arbitrator_output.arbitration_confidence


# ============================================================================
# Ground-truth-aware disagreement resolution (src/disagreement.py)
#
# These texts are hand-picked, not arbitrary — under the current mock
# critic hashing scheme (src/critics.py), each one deterministically
# produces a real flag/no-flag split on a specific dimension. That's a
# precondition for these tests to mean anything: if no split occurs, an
# assertion like "no directional label appeared" would pass vacuously
# without actually testing the resolution logic. If critics.py's mock
# logic ever changes, these texts may need to be re-picked.
# ============================================================================

_SPLIT_TEXT_FACTUAL = "Test claim number 0 about knuckle cracking causing arthritis."
_SPLIT_TEXT_REASONING = "Reasoning test output number 0 explaining a multi step process."


def test_disagreement_resolution_without_ground_truth_stays_neutral():
    """No benchmark_item passed -> a real issue-presence split on
    factual_accuracy must NOT assert a direction, since there's nothing to
    check it against."""
    result = run_pipeline(input_id="gt_neutral", original_output=_SPLIT_TEXT_FACTUAL, task_type="factual_qa")

    factual_events = [e for e in result.disagreement_matrix.events if "factual_accuracy" in e.description]
    split_events = [e for e in factual_events if e.disagreement_type == "issue_presence_split"]
    directional_events = [e for e in factual_events if e.disagreement_type in ("issue_detection_miss", "false_positive_flag")]

    assert len(split_events) == 1, "expected exactly one unresolved split on factual_accuracy"
    assert len(directional_events) == 0, "no ground truth was provided — nothing should assert a direction"


def test_disagreement_resolution_with_untruthful_ground_truth():
    """TruthfulQA 'untruthful' = ground truth confirms a real issue exists.
    The critic that did NOT flag it should be labeled as having missed a
    real issue, not left neutral."""
    item = BenchmarkItem(
        item_id="gt_untruthful",
        dataset_source=DatasetSource.TRUTHFULQA,
        task_type=TaskType.FACTUAL_QA,
        input_prompt="Does cracking your knuckles cause arthritis?",
        output_text=_SPLIT_TEXT_FACTUAL,
        ground_truth_label="untruthful",
        human_quality_score_basis=QualityScoreBasis.BINARY_TRUTHFUL_LABEL,
    )
    result = run_pipeline(input_id=item.item_id, original_output=_SPLIT_TEXT_FACTUAL, task_type="factual_qa", benchmark_item=item)

    factual_events = [e for e in result.disagreement_matrix.events if "factual_accuracy" in e.description]
    miss_events = [e for e in factual_events if e.disagreement_type == "issue_detection_miss"]
    split_events = [e for e in factual_events if e.disagreement_type == "issue_presence_split"]

    assert len(miss_events) == 1, "ground truth confirms a real issue — the non-flagging critic should be marked as having missed it"
    assert len(split_events) == 0, "should not stay neutral once ground truth resolves the direction"
    assert "critic_a" in miss_events[0].critics_involved


def test_disagreement_resolution_with_truthful_ground_truth():
    """TruthfulQA 'truthful' = ground truth confirms NO issue exists. The
    critics that DID flag one should be labeled as a false positive."""
    item = BenchmarkItem(
        item_id="gt_truthful",
        dataset_source=DatasetSource.TRUTHFULQA,
        task_type=TaskType.FACTUAL_QA,
        input_prompt="Does cracking your knuckles cause arthritis?",
        output_text=_SPLIT_TEXT_FACTUAL,
        ground_truth_label="truthful",
        human_quality_score_basis=QualityScoreBasis.BINARY_TRUTHFUL_LABEL,
    )
    result = run_pipeline(input_id=item.item_id, original_output=_SPLIT_TEXT_FACTUAL, task_type="factual_qa", benchmark_item=item)

    factual_events = [e for e in result.disagreement_matrix.events if "factual_accuracy" in e.description]
    fp_events = [e for e in factual_events if e.disagreement_type == "false_positive_flag"]
    split_events = [e for e in factual_events if e.disagreement_type == "issue_presence_split"]

    assert len(fp_events) == 1, "ground truth confirms no issue exists — the flagging critics should be marked false positive"
    assert len(split_events) == 0
    assert set(fp_events[0].critics_involved) == {"critic_b", "critic_c"}


def test_disagreement_resolution_ignores_non_factual_dimensions():
    """Ground truth only ever covers factual_accuracy. A real split on
    logical_consistency must stay neutral even with a benchmark_item
    present, since there's no ground truth signal for that dimension."""
    item = BenchmarkItem(
        item_id="gt_wrong_dimension",
        dataset_source=DatasetSource.TRUTHFULQA,
        task_type=TaskType.REASONING,
        input_prompt="x",
        output_text=_SPLIT_TEXT_REASONING,
        ground_truth_label="untruthful",
        human_quality_score_basis=QualityScoreBasis.BINARY_TRUTHFUL_LABEL,
    )
    result = run_pipeline(input_id=item.item_id, original_output=_SPLIT_TEXT_REASONING, task_type="reasoning", benchmark_item=item)

    logic_events = [e for e in result.disagreement_matrix.events if "logical_consistency" in e.description]
    split_events = [e for e in logic_events if e.disagreement_type == "issue_presence_split"]
    directional_events = [e for e in logic_events if e.disagreement_type in ("issue_detection_miss", "false_positive_flag")]

    assert len(split_events) == 1, "expected a real split on logical_consistency for this fixture text"
    assert len(directional_events) == 0, "ground truth resolution must never fire outside factual_accuracy"


# ============================================================================
# Critic failure handling (src/critics.py's make_failed_critique,
# src/disagreement.py's _successful filtering)
# ============================================================================

def test_pipeline_survives_a_single_critic_failure():
    """One critic raising should not crash the pipeline — it should be
    marked failed and the pipeline should still produce a valid result
    using the other two."""
    original_evaluate = CRITICS["critic_b"].evaluate

    def _boom(self, output_text, dimensions):
        raise TimeoutError("simulated 30s timeout")

    with patch.object(type(CRITICS["critic_b"]), "evaluate", _boom):
        result = run_pipeline(input_id="failure_test", original_output="A short test output for failure handling.", task_type="factual_qa")

    assert isinstance(result, ArbitrationResult)
    assert len(result.critiques) == 3  # all 3 slots still present

    failed = [c for c in result.critiques if c.critic_failed]
    succeeded = [c for c in result.critiques if not c.critic_failed]
    assert len(failed) == 1
    assert failed[0].critic_id == "critic_b"
    assert "simulated 30s timeout" in failed[0].failure_reason
    assert failed[0].self_confidence is None
    assert failed[0].dimension_scores == {}
    assert len(succeeded) == 2

    # ML features must reflect only the 2 successful critics, not 3
    assert result.ml_features.critics_used_count == 2

    # Disagreement detection must not crash or misclassify the failed
    # critic as "checked this dimension and found nothing"
    for event in result.disagreement_matrix.events:
        assert "critic_b" not in event.critics_involved

    # Restore for any other test that might run in the same process
    CRITICS["critic_b"].evaluate = original_evaluate.__get__(CRITICS["critic_b"], type(CRITICS["critic_b"]))


def test_routed_dimensions_uses_first_successful_critic_not_index_zero():
    """Regression test for the exact bug this fix corrects: if the FIRST
    critic in iteration order fails, disagreement detection must still run
    using a successful critic's dimensions — not silently return zero
    dimensions because critiques[0] happened to be the failed one."""
    def _boom(self, output_text, dimensions):
        raise RuntimeError("simulated API error")

    with patch.object(type(CRITICS["critic_a"]), "evaluate", _boom):
        result = run_pipeline(input_id="first_critic_fails", original_output="The Eiffel Tower was built in 1889 in Paris, France.", task_type="factual_qa")

    # factual_qa routes 2 dimensions — disagreement detection should still
    # have run against them using critic_b/critic_c, not returned nothing
    # because critic_a (first in the dict) failed.
    assert result.ml_features.critics_used_count == 2
    critic_a_result = next(c for c in result.critiques if c.critic_id == "critic_a")
    assert critic_a_result.critic_failed is True