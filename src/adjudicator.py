"""
src/adjudicator.py

Real gpt-5.6-terra adjudicator. Runs when arbitrator.py escalates.
Reviews every flagged issue and either confirms or dismisses it with a
reason, then gives its own final quality score.

reasoning_effort="none" is required -- gpt-5.6-terra is a reasoning-tier
model, and OpenAI's chat completions endpoint rejects function-tool calls
(how instructor implements response_model=) combined with reasoning_effort
otherwise. temperature is dropped for the same family-of-model reason --
reasoning models commonly reject custom temperature. Flagging this one as
not fully confirmed -- if you see a new 400 about temperature specifically,
that's why, tell me immediately.

Evidence retrieval is still a placeholder -- ChromaDB isn't built. Always [].
"""

from typing import Optional

from pydantic import BaseModel

from src.critics import _get_openai_client
from src.schemas import CritiqueOutput, DisagreementMatrix, MLArbitratorOutput, Verdict, Issue, DismissedFlag

ADJUDICATOR_MODEL = "gpt-5.6-terra"
MAX_OUTPUT_CHARS = 6000


def retrieve_evidence(original_output: str) -> list[str]:
    return []


class IssueVerdict(BaseModel):
    issue_index: int
    keep: bool
    reason: str


class AdjudicatorResponse(BaseModel):
    quality_score: int  # 1-10
    confidence: int  # 1-5
    issue_verdicts: list[IssueVerdict]
    reasoning: str


def _build_prompt(
    original_output: str,
    indexed_issues: list[tuple[int, str, Issue]],
    disagreement_matrix: DisagreementMatrix,
    ml_output: MLArbitratorOutput,
) -> str:
    output_snippet = original_output[:MAX_OUTPUT_CHARS]

    issue_lines = "\n".join(
        f'{idx}. [{critic_id}, {issue.severity}] {issue.description} (quote: "{issue.quoted_evidence}")'
        for idx, critic_id, issue in indexed_issues
    ) or "(no issues flagged)"

    return f"""You are the final adjudicator. 3 critics evaluated this response and either disagreed or the arbitrator wasn't confident. Review each flagged issue and decide if it's real.

Response being evaluated:
\"\"\"
{output_snippet}
\"\"\"

Issues flagged:
{issue_lines}

Disagreement: {disagreement_matrix.disagreement_count} event(s), max score gap {disagreement_matrix.max_score_gap}.
ML arbitrator's estimate: quality {ml_output.predicted_quality_score}/10, confidence {ml_output.arbitration_confidence}.

For each issue, keep (real problem) or dismiss (false alarm), with a short reason. Then give your own final quality_score (1-10) and confidence (1-5).

Return JSON only:
{{
  "quality_score": <1-10 int>,
  "confidence": <1-5 int>,
  "issue_verdicts": [{{"issue_index": int, "keep": bool, "reason": str}}, ...],
  "reasoning": str
}}
"""


def run_adjudicator(
    original_output: str,
    critiques: list[CritiqueOutput],
    disagreement_matrix: DisagreementMatrix,
    ml_output: MLArbitratorOutput,
) -> tuple[Verdict, Optional[int], Optional[int]]:
    """Returns (Verdict, input_tokens, output_tokens)."""
    indexed_issues: list[tuple[int, str, Issue]] = []
    i = 0
    for c in critiques:
        for issue in c.issues:
            indexed_issues.append((i, c.critic_id, issue))
            i += 1

    prompt = _build_prompt(original_output, indexed_issues, disagreement_matrix, ml_output)

    result, completion = _get_openai_client().chat.completions.create_with_completion(
        model=ADJUDICATOR_MODEL,
        response_model=AdjudicatorResponse,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="none",
    )

    issue_by_index = {idx: (cid, issue) for idx, cid, issue in indexed_issues}
    confirmed_issues: list[Issue] = []
    dismissed_flags: list[DismissedFlag] = []

    for v in result.issue_verdicts:
        lookup = issue_by_index.get(v.issue_index)
        if lookup is None:
            continue
        cid, issue = lookup
        if v.keep:
            confirmed_issues.append(issue)
        else:
            dismissed_flags.append(DismissedFlag(
                description=issue.description, raised_by=cid, dismissal_reason=v.reason,
            ))

    verdict = Verdict(
        quality_score=max(1, min(10, result.quality_score)),
        confidence=max(1, min(5, result.confidence)),
        confirmed_issues=confirmed_issues,
        dismissed_flags=dismissed_flags,
        adjudicated=True,
        adjudicated_by="gpt4o_adjudicator",
        adjudicator_reasoning=result.reasoning,
    )

    return verdict, completion.usage.prompt_tokens, completion.usage.completion_tokens