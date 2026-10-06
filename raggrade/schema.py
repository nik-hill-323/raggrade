"""Data model for evaluation cases and results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Passage(BaseModel):
    id: str
    text: str


class EvalCase(BaseModel):
    """One question run through a RAG system, plus what it should have done.

    ``relevant_ids`` and ``reference_answer`` are optional: without them you
    still get faithfulness, answer relevance and citation coverage, which need
    no labels at all.
    """

    id: str
    question: str
    answer: str
    retrieved: list[Passage] = Field(default_factory=list)
    relevant_ids: list[str] = Field(default_factory=list)
    reference_answer: str | None = None


class EvalResult(BaseModel):
    id: str
    # retrieval (only when relevant_ids given)
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    mrr: float | None = None
    ndcg_at_k: float | None = None
    # generation
    faithfulness: float
    answer_relevance: float
    citation_coverage: float
    unsupported_claims: list[str] = Field(default_factory=list)
    # labels
    reference_similarity: float | None = None


class EvalSummary(BaseModel):
    n: int
    judge: str
    means: dict[str, float]
    results: list[EvalResult]

    def table(self) -> str:
        rows = ["metric                  mean"]
        for k, v in self.means.items():
            rows.append(f"{k:<24}{v:.3f}")
        return "\n".join(rows)
