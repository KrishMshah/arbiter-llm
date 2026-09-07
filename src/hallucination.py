"""
src/hallucination.py

Sentence-level hallucination tracer. Fully deterministic — no LLM calls,
just text matching and simple heuristics, per guardrails spec.

Sentence splitting uses nltk's punkt tokenizer (trained to handle
abbreviations like "Dr.", "U.S.", "etc." correctly — regex alone gets
these wrong and corrupts origin-sentence matching). Falls back to a
regex splitter automatically if punkt data isn't downloaded on a given
machine, so this never hard-fails across the team's 3 setups.

Origin detection only looks at factual_accuracy issues (hallucination is
a factual-accuracy concept per the project doc's Layer 5 description).
Dependency-chain heuristics are intentionally simple (pronoun references +
discourse markers + keyword overlap) per the project doc's explicit
instruction: no LLM calls, keep it deterministic and cheap — the research
value is in the dataset of traces, not the perfection of each trace.
"""

import re
from src.schemas import CritiqueOutput, HallucinationTrace, OriginSentence, DependentSentence

MAX_PROPAGATION_DEPTH = 10  # guardrail: prevent runaway chains

_REGEX_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')

_PRONOUNS = {"it", "its", "they", "their", "them", "this", "that", "these", "those"}
_DISCOURSE_MARKERS = [
    "therefore", "furthermore", "as a result", "this means", "moreover",
    "consequently", "thus", "hence", "in addition", "because of this",
]

# Checked once at import time so every call doesn't re-attempt the import.
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    sent_tokenize("Warm-up call to confirm punkt data is present.")
    _NLTK_AVAILABLE = True
except (ImportError, LookupError):
    _NLTK_AVAILABLE = False


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if _NLTK_AVAILABLE:
        return [s.strip() for s in sent_tokenize(text) if s.strip()]

    return [s.strip() for s in _REGEX_SENTENCE_SPLIT.split(text) if s.strip()]


def _find_origin_sentences(sentences: list[str], critiques: list[CritiqueOutput]) -> dict[int, OriginSentence]:
    """Cross-reference factual_accuracy issue evidence against sentences.
    A sentence containing the quoted evidence text becomes an origin.
    Multiple critics flagging the same sentence just adds to flagged_by."""
    origins: dict[int, OriginSentence] = {}

    for critique in critiques:
        for issue in critique.issues:
            if issue.dimension != "factual_accuracy":
                continue
            evidence = issue.quoted_evidence.strip()
            if not evidence:
                continue
            for idx, sentence in enumerate(sentences):
                if evidence in sentence:
                    if idx in origins:
                        if critique.critic_id not in origins[idx].flagged_by:
                            origins[idx].flagged_by.append(critique.critic_id)
                    else:
                        origins[idx] = OriginSentence(
                            sentence_index=idx,
                            text=sentence,
                            flagged_by=[critique.critic_id],
                            issue_description=issue.description,
                        )
    return origins


def _extract_keywords(sentence: str) -> set[str]:
    """Crude keyword set: lowercased words 4+ chars. Not NLP-grade — just
    enough signal to catch shared entities/terms between sentences."""
    return set(re.findall(r"[A-Za-z]{4,}", sentence.lower()))


def _has_pronoun_reference(sentence: str) -> bool:
    words = re.findall(r"[a-z']+", sentence.lower())
    return any(w in _PRONOUNS for w in words)


def _has_discourse_marker(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(marker in lowered for marker in _DISCOURSE_MARKERS)


def _find_dependents(sentences: list[str], origins: dict[int, OriginSentence]) -> list[DependentSentence]:
    dependents: list[DependentSentence] = []

    for origin_idx in sorted(origins.keys()):
        origin_keywords = _extract_keywords(sentences[origin_idx])
        chain_depth = 0

        for idx in range(origin_idx + 1, len(sentences)):
            if idx in origins:
                break  # next origin starts its own chain
            if chain_depth >= MAX_PROPAGATION_DEPTH:
                break

            sentence = sentences[idx]
            dep_type = None

            if _has_pronoun_reference(sentence):
                dep_type = "pronoun_reference"
            elif _has_discourse_marker(sentence):
                dep_type = "logical_extension"
            else:
                overlap = _extract_keywords(sentence) & origin_keywords
                if len(overlap) >= 2:
                    dep_type = "factual_reference"

            if dep_type is None:
                break  # chain broken — this sentence doesn't reference the origin

            dependents.append(DependentSentence(
                sentence_index=idx,
                text=sentence,
                depends_on_origin_index=origin_idx,
                dependency_type=dep_type,
            ))
            chain_depth += 1

    return dependents


def trace_hallucinations(output_text: str, critiques: list[CritiqueOutput]) -> HallucinationTrace:
    sentences = _split_sentences(output_text)

    if len(sentences) < 2:
        return HallucinationTrace(
            has_hallucination=False, origin_sentences=[], dependent_sentences=[],
            propagation_depth=0, total_affected_sentences=0,
        )

    origins = _find_origin_sentences(sentences, critiques)
    if not origins:
        return HallucinationTrace(
            has_hallucination=False, origin_sentences=[], dependent_sentences=[],
            propagation_depth=0, total_affected_sentences=0,
        )

    dependents = _find_dependents(sentences, origins)

    depth_per_origin: dict[int, int] = {}
    for dep in dependents:
        depth_per_origin[dep.depends_on_origin_index] = depth_per_origin.get(dep.depends_on_origin_index, 0) + 1
    propagation_depth = max(depth_per_origin.values(), default=0)

    return HallucinationTrace(
        has_hallucination=True,
        origin_sentences=list(origins.values()),
        dependent_sentences=dependents,
        propagation_depth=propagation_depth,
        total_affected_sentences=len(origins) + len(dependents),
    )