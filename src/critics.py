"""
src/critics.py

BaseCritic + 3 critic implementations. Phase 2: all three real.
CriticC = Ollama, CriticA = OpenAI, CriticB = Anthropic, via instructor.
"""

import hashlib
import json
import random
from abc import ABC, abstractmethod

import instructor
import ollama  # type: ignore
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

from src.config import (
    ANTHROPIC_API_KEY,
    MOCK_CRITIC_A,
    MOCK_CRITIC_B,
    MOCK_CRITIC_C,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
)
from src.schemas import CritiqueOutput, Issue, Severity


def _seeded_rng(output_text: str, critic_id: str, salt: str = "") -> random.Random:
    seed = int(hashlib.sha256(f"{critic_id}:{salt}:{output_text}".encode()).hexdigest(), 16)
    return random.Random(seed)


def make_failed_critique(critic_id: str, model_used: str, reason: str) -> CritiqueOutput:
    return CritiqueOutput(
        critic_id=critic_id,
        model_used=model_used,
        dimension_scores={},
        issues=[],
        self_confidence=None,
        reasoning=f"[FAILED] {reason}",
        dimensions_evaluated=[],
        critic_failed=True,
        failure_reason=reason,
    )


DIMENSION_DESCRIPTIONS = {
    "factual_accuracy": "Is every factual claim correct and verifiable?",
    "logical_consistency": "Does the reasoning hold together without contradiction?",
    "completeness": "Does the response fully address what was asked?",
}


def build_prompt(output_text: str, dimensions: list[str]) -> str:
    dim_lines = "\n".join(f"- {d}: {DIMENSION_DESCRIPTIONS.get(d, d)}" for d in dimensions)
    return f"""You are an expert evaluator. Score the response on each dimension below, 1 (worst) to 5 (best). Flag specific issues with a quoted excerpt and severity.

Dimensions:
{dim_lines}

Response to evaluate:
\"\"\"
{output_text}
\"\"\"

Return JSON only, matching this shape:
{{
  "dimension_scores": {{"<dimension>": <1-5 int>, ...}},
  "issues": [{{"description": str, "quoted_evidence": str, "severity": "minor"|"major"|"critical", "dimension": str}}],
  "self_confidence": <1-5 int>,
  "reasoning": str
}}
"""


def build_critique_schema(dimensions: list[str]) -> dict:
    # additionalProperties: False stops the model inventing dimensions/keys
    return {
        "type": "object",
        "properties": {
            "dimension_scores": {
                "type": "object",
                "properties": {d: {"type": "integer", "minimum": 1, "maximum": 5} for d in dimensions},
                "required": dimensions,
                "additionalProperties": False,
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quoted_evidence": {"type": "string"},
                        "severity": {"type": "string", "enum": ["minor", "major", "critical"]},
                        "dimension": {"type": "string", "enum": dimensions},
                    },
                    "required": ["description", "quoted_evidence", "severity", "dimension"],
                },
            },
            "self_confidence": {"type": "integer", "minimum": 1, "maximum": 5},
            "reasoning": {"type": "string"},
        },
        "required": ["dimension_scores", "issues", "self_confidence", "reasoning"],
    }


class CritiqueResponse(BaseModel):
    """What instructor asks GPT-4o-mini / Claude Haiku to return."""
    dimension_scores: dict[str, int]
    issues: list[Issue]
    self_confidence: int
    reasoning: str


_openai_client = None
_anthropic_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = instructor.from_openai(OpenAI(api_key=OPENAI_API_KEY))
    return _openai_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = instructor.from_anthropic(Anthropic(api_key=ANTHROPIC_API_KEY))
    return _anthropic_client


def _to_critique_output(critic_id: str, model_used: str, result: CritiqueResponse, dimensions: list[str]) -> CritiqueOutput:
    # instructor doesn't schema-lock keys like Ollama's format= does — filter defensively
    extra = set(result.dimension_scores) - set(dimensions)
    if extra:
        print(f"[{critic_id}] warning: dropped unexpected dimensions {extra}")

    return CritiqueOutput(
        critic_id=critic_id,
        model_used=model_used,
        dimension_scores={d: s for d, s in result.dimension_scores.items() if d in dimensions},
        issues=[i for i in result.issues if i.dimension in dimensions],
        self_confidence=result.self_confidence,
        reasoning=result.reasoning,
        dimensions_evaluated=dimensions,
    )


class BaseCritic(ABC):
    critic_id: str
    model_used: str

    @abstractmethod
    def evaluate(self, output_text: str, dimensions: list[str]) -> CritiqueOutput:
        ...

    def _mock_evaluate(self, output_text: str, dimensions: list[str]) -> CritiqueOutput:
        dimension_scores: dict[str, int] = {}
        issues: list[Issue] = []

        for dim in dimensions:
            rng = _seeded_rng(output_text, self.critic_id, salt=dim)
            score = rng.randint(1, 5)
            dimension_scores[dim] = score
            if score <= 3:
                issues.append(Issue(
                    description=f"Mock-detected {dim} concern",
                    quoted_evidence=output_text[:40] or "(empty output)",
                    severity=rng.choice(list(Severity)),
                    dimension=dim,
                ))

        confidence_rng = _seeded_rng(output_text, self.critic_id, salt="confidence")
        confidence = confidence_rng.randint(1, 5)

        return CritiqueOutput(
            critic_id=self.critic_id,
            model_used=self.model_used,
            dimension_scores=dimension_scores,
            issues=issues,
            self_confidence=confidence,
            reasoning=f"[MOCK] {self.critic_id} evaluated {', '.join(dimensions)} deterministically.",
            dimensions_evaluated=dimensions,
        )


class CriticA(BaseCritic):
    critic_id = "critic_a"
    model_used = "gpt-4o-mini"

    def evaluate(self, output_text: str, dimensions: list[str]) -> CritiqueOutput:
        if MOCK_CRITIC_A:
            return self._mock_evaluate(output_text, dimensions)

        result = _get_openai_client().chat.completions.create(
            model=self.model_used,
            response_model=CritiqueResponse,
            messages=[{"role": "user", "content": build_prompt(output_text, dimensions)}],
            temperature=0.2,
        )
        return _to_critique_output(self.critic_id, self.model_used, result, dimensions)


class CriticB(BaseCritic):
    critic_id = "critic_b"
    model_used = "claude-haiku-4-5"

    def evaluate(self, output_text: str, dimensions: list[str]) -> CritiqueOutput:
        if MOCK_CRITIC_B:
            return self._mock_evaluate(output_text, dimensions)

        result = _get_anthropic_client().messages.create(
            model=self.model_used,
            response_model=CritiqueResponse,
            max_tokens=2048,  # required by Anthropic's API, unlike OpenAI
            messages=[{"role": "user", "content": build_prompt(output_text, dimensions)}],
            temperature=0.2,
        )
        return _to_critique_output(self.critic_id, self.model_used, result, dimensions)


class CriticC(BaseCritic):
    critic_id = "critic_c"

    @property
    def model_used(self) -> str:
        return OLLAMA_MODEL  # read at call time so base->fine-tuned swap needs no code change

    def evaluate(self, output_text: str, dimensions: list[str]) -> CritiqueOutput:
        if MOCK_CRITIC_C:
            return self._mock_evaluate(output_text, dimensions)

        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": build_prompt(output_text, dimensions)}],
            format=build_critique_schema(dimensions),
            options={"temperature": 0.2},
        )
        parsed = json.loads(response["message"]["content"])
        issues = [Issue(**i) for i in parsed.get("issues", [])]

        return CritiqueOutput(
            critic_id=self.critic_id,
            model_used=self.model_used,
            dimension_scores=parsed["dimension_scores"],
            issues=issues,
            self_confidence=parsed.get("self_confidence"),
            reasoning=parsed.get("reasoning", ""),
            dimensions_evaluated=dimensions,
        )


CRITICS: dict[str, BaseCritic] = {
    "critic_a": CriticA(),
    "critic_b": CriticB(),
    "critic_c": CriticC(),
}