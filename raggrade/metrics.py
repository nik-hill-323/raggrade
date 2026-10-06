"""Metric implementations.

Retrieval metrics follow the usual information-retrieval definitions and need
only ids. Generation metrics decompose the answer into sentences and ask a
judge about each one, which is the same shape as RAGAS faithfulness but with
the judge swappable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .judges import Judge
from .text import citations, sentences, strip_citations, tokens


@dataclass
class RetrievalScores:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


def retrieval_scores(retrieved_ids: list[str], relevant_ids: list[str]) -> RetrievalScores:
    """Precision@k, recall@k and nDCG@k over the retrieved list (k = its
    length), and reciprocal rank of the first relevant hit."""
    if not retrieved_ids:
        return RetrievalScores(0.0, 0.0, 0.0, 0.0)
    relevant = set(relevant_ids)
    hits = [rid in relevant for rid in retrieved_ids]
    precision = sum(hits) / len(retrieved_ids)
    recall = sum(hits) / len(relevant) if relevant else 0.0
    mrr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / rank
            break
    return RetrievalScores(precision, recall, mrr, ndcg_at_k(retrieved_ids, relevant_ids))


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int | None = None) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    Unlike precision and recall it rewards putting the relevant passages near
    the top: a hit at rank i contributes 1 / log2(i + 1), and the total is
    divided by the gain of the ideal ranking, where every relevant passage
    (up to k of them) comes first. ``k`` defaults to the retrieved length.
    Duplicate ids only count the first time, so a retriever cannot score by
    repeating a relevant passage.
    """
    relevant = set(relevant_ids)
    k = len(retrieved_ids) if k is None else k
    if k <= 0 or not relevant:
        return 0.0
    seen: set[str] = set()
    dcg = 0.0
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant and rid not in seen:
            dcg += 1.0 / math.log2(rank + 1)
        seen.add(rid)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal


def faithfulness(answer: str, passages: list[str], judge: Judge) -> tuple[float, list[str]]:
    """Fraction of answer sentences the judge finds supported by the passages,
    plus the unsupported sentences so you can read them."""
    sents = [strip_citations(s) for s in sentences(answer)]
    sents = [s for s in sents if tokens(s)]
    if not sents:
        return 1.0, []
    if not passages:
        return 0.0, sents
    unsupported = [s for s in sents if not judge.supports(s, passages)]
    return 1 - len(unsupported) / len(sents), unsupported


def answer_relevance(question: str, answer: str, judge: Judge) -> float:
    return judge.relevance(question, answer)


def citation_coverage(answer: str, n_passages: int) -> float:
    """Fraction of sentences that carry at least one citation marker like [2]
    that points at an actual retrieved passage. Systems that promise cited
    output should score 1.0 here; the metric says nothing about whether the
    citation is the *right* one, which is what faithfulness is for."""
    sents = [s for s in sentences(answer) if tokens(strip_citations(s))]
    if not sents:
        return 1.0
    ok = 0
    for s in sents:
        refs = citations(s)
        if refs and all(1 <= r <= n_passages for r in refs):
            ok += 1
    return ok / len(sents)


def reference_similarity(answer: str, reference: str) -> float:
    """Symmetric content-word F1 between the answer and a reference answer.
    Cheap, and correlates well enough with human judgement for regression
    tracking, which is what it is for."""
    a, r = tokens(answer), tokens(reference)
    if not a or not r:
        return 0.0
    common = len(set(a) & set(r))
    if common == 0:
        return 0.0
    p, rec = common / len(set(a)), common / len(set(r))
    return 2 * p * rec / (p + rec)
