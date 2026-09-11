"""Small text utilities shared by the metrics and the offline judge."""

from __future__ import annotations

import re

# Split after sentence punctuation, or after a trailing citation marker, but only
# when the next character starts a new sentence, so "30. [1] It" becomes
# ["30. [1]", "It ..."] rather than splitting the marker off on its own.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|(?<=\])\s+(?=[A-Z0-9])")
_CITATION = re.compile(r"\[(\d+)\]")
_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    ["a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in", "on", "at", "by", "for", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "there", "here", "which", "who", "whom", "what", "when", "where", "why", "how", "not", "no", "nor", "do", "does", "did", "done", "has", "have", "had", "having", "can", "could", "may", "might", "shall", "should", "will", "would", "than", "too", "very", "so", "such", "into", "onto", "over", "under", "about", "above", "below", "between", "through"]
)


def sentences(text: str) -> list[str]:
    """Split on sentence boundaries. Keeps citation markers attached to the sentence."""
    parts = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    return parts


def citations(sentence: str) -> list[int]:
    return [int(m) for m in _CITATION.findall(sentence)]


def strip_citations(sentence: str) -> str:
    return _CITATION.sub("", sentence).strip()


def tokens(text: str, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def overlap(a: str, b: str) -> float:
    """Fraction of content tokens in ``a`` that also appear in ``b``."""
    ta = tokens(a)
    if not ta:
        return 1.0
    tb = set(tokens(b))
    return sum(t in tb for t in ta) / len(ta)
