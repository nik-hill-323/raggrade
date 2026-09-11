"""ragcheck: evaluation for retrieval-augmented generation systems.

A RAG answer can fail in three places. Retrieval can miss the passage that
holds the answer. Generation can say things the retrieved passages do not
support. And the answer can be faithful to its sources but not actually answer
the question. ragcheck scores each failure mode separately so you can see
which part of the pipeline to fix.

Retrieval metrics need only the ids of retrieved and relevant passages.
Generation metrics work with a pluggable judge: the offline judge is
deterministic and dependency-free, so it runs in CI; the LLM judges (Anthropic
or OpenAI) are more accurate and are meant for offline evaluation runs.
"""

from .judges import Judge, LexicalJudge
from .metrics import (
    RetrievalScores,
    citation_coverage,
    retrieval_scores,
)
from .runner import evaluate
from .schema import EvalCase, EvalResult, EvalSummary

__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalSummary",
    "Judge",
    "LexicalJudge",
    "RetrievalScores",
    "citation_coverage",
    "evaluate",
    "retrieval_scores",
]

__version__ = "0.1.0"
