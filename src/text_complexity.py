"""Utility functions for computing text complexity signals (readability and lexical diversity)."""

from __future__ import annotations

import re
from typing import Optional, Sequence

try:  # pragma: no cover
    from dart.handler.other.textstat import TextStatHandler
except ImportError:  # pragma: no cover
    TextStatHandler = None  # type: ignore[misc,assignment]

WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def _tokenize(text: str) -> Sequence[str]:
    if not text:
        return []
    return WORD_PATTERN.findall(text.lower())


def lix_score(text: str) -> float:
    """
    Compute the LIX readability score.

    LIX = (number of words / number of sentences) + (number of long words * 100 / number of words)
    """
    if not text:
        return 0.0

    sentences = re.split(r"[\.!?]+", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    num_sentences = max(len(sentences), 1)

    words = WORD_PATTERN.findall(text)
    num_words = max(len(words), 1)
    num_long_words = sum(1 for word in words if len(word) > 6)
    return (num_words / num_sentences) + (num_long_words * 100.0 / num_words)


def flesch_kincaid_score(
    text: str,
    scorer: Optional["TextStatHandler"] = None,
    *,
    language: str = "english",
) -> float:
    """Compute Flesch–Kincaid readability, falling back to zero if no scorer is available."""
    if not text:
        return 0.0

    fk_scorer = scorer
    if fk_scorer is None and TextStatHandler is not None:
        try:
            fk_scorer = TextStatHandler(language)
        except Exception:
            fk_scorer = None

    if fk_scorer is None:
        return 0.0

    try:
        return float(fk_scorer.flesch_kincaid_score(text))
    except Exception:
        return 0.0


def _mtld_direction(tokens: Sequence[str], threshold: float) -> float:
    """
    Compute the directional MTLD contribution for a token sequence.

    Implementation based on: McCarthy & Jarvis (2010), with partial factor handling.
    """
    if not tokens:
        return 0.0
    if not 0 < threshold < 1:
        raise ValueError("MTLD threshold must be between 0 and 1.")

    type_counts = {}
    token_count = 0
    factor_count = 0.0

    for token in tokens:
        token_count += 1
        type_counts[token] = type_counts.get(token, 0) + 1
        ttr = len(type_counts) / token_count
        if ttr <= threshold:
            factor_count += 1.0
            type_counts.clear()
            token_count = 0

    if token_count > 0:
        ttr = len(type_counts) / token_count if token_count else 1.0
        if ttr <= 0:
            ttr = threshold
        factor_count += (1 - ttr) / (1 - threshold)

    if factor_count <= 0:
        return float(len(tokens))

    return float(len(tokens)) / factor_count


def mtld_score(text: str, threshold: float = 0.72) -> float:
    """Compute the Measure of Textual Lexical Diversity (MTLD)."""
    tokens = list(_tokenize(text))
    if not tokens:
        return 0.0

    forward = _mtld_direction(tokens, threshold)
    backward = _mtld_direction(list(reversed(tokens)), threshold)
    if forward <= 0 and backward <= 0:
        return 0.0
    if forward <= 0:
        return backward
    if backward <= 0:
        return forward
    return 0.5 * (forward + backward)


__all__ = ["lix_score", "flesch_kincaid_score", "mtld_score"]
