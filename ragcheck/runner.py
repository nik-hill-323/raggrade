"""Run every metric over a list of cases and aggregate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .judges import Judge, LexicalJudge
from .metrics import (
    answer_relevance,
    citation_coverage,
    faithfulness,
    reference_similarity,
    retrieval_scores,
)
from .schema import EvalCase, EvalResult, EvalSummary


def evaluate_case(case: EvalCase, judge: Judge) -> EvalResult:
    passages = [p.text for p in case.retrieved]
    faith, unsupported = faithfulness(case.answer, passages, judge)
    result = EvalResult(
        id=case.id,
        faithfulness=faith,
        answer_relevance=answer_relevance(case.question, case.answer, judge),
        citation_coverage=citation_coverage(case.answer, len(passages)),
        unsupported_claims=unsupported,
    )
    if case.relevant_ids:
        r = retrieval_scores([p.id for p in case.retrieved], case.relevant_ids)
        result.precision_at_k, result.recall_at_k, result.mrr = r.precision_at_k, r.recall_at_k, r.mrr
    if case.reference_answer:
        result.reference_similarity = reference_similarity(case.answer, case.reference_answer)
    return result


def evaluate(cases: list[EvalCase], judge: Judge | None = None) -> EvalSummary:
    judge = judge or LexicalJudge()
    results = [evaluate_case(c, judge) for c in cases]
    means: dict[str, float] = {}
    for field in (
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "faithfulness",
        "answer_relevance",
        "citation_coverage",
        "reference_similarity",
    ):
        vals = [getattr(r, field) for r in results if getattr(r, field) is not None]
        if vals:
            means[field] = float(np.mean(vals))
    return EvalSummary(n=len(results), judge=judge.name, means=means, results=results)


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load cases from a JSON list or a JSONL file."""
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [EvalCase.model_validate_json(line) for line in text.splitlines() if line.strip()]
    return [EvalCase.model_validate(obj) for obj in json.loads(text)]
