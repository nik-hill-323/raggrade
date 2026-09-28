import pytest

from raggrade import LexicalJudge, citation_coverage, retrieval_scores
from raggrade.metrics import faithfulness, reference_similarity

PASSAGES = [
    "Metformin is contraindicated in severe renal impairment (eGFR below 30) and in metabolic acidosis.",
    "Metformin remains first-line therapy for type 2 diabetes.",
]


def test_retrieval_scores_basic():
    r = retrieval_scores(["a", "b", "c", "d"], ["b", "d", "z"])
    assert r.precision_at_k == pytest.approx(0.5)
    assert r.recall_at_k == pytest.approx(2 / 3)
    assert r.mrr == pytest.approx(0.5)


def test_retrieval_scores_empty():
    r = retrieval_scores([], ["a"])
    assert (r.precision_at_k, r.recall_at_k, r.mrr) == (0.0, 0.0, 0.0)


def test_faithful_answer_scores_one():
    score, bad = faithfulness(
        "Metformin is contraindicated in severe renal impairment. It is first-line for type 2 diabetes.",
        PASSAGES,
        LexicalJudge(),
    )
    assert score == 1.0
    assert bad == []


def test_hallucinated_sentence_is_caught_and_returned():
    score, bad = faithfulness(
        "Metformin is contraindicated in severe renal impairment. Metformin cures diabetes in two weeks.",
        PASSAGES,
        LexicalJudge(),
    )
    assert score == 0.5
    assert bad == ["Metformin cures diabetes in two weeks."]


def test_no_passages_means_nothing_is_supported():
    score, bad = faithfulness("Anything at all.", [], LexicalJudge())
    assert score == 0.0 and bad == ["Anything at all."]


def test_citation_coverage_counts_valid_markers_only():
    assert citation_coverage("A fact. [1] Another fact. [2]", n_passages=2) == 1.0
    assert citation_coverage("A fact. [1] Another fact.", n_passages=2) == 0.5
    assert citation_coverage("A fact. [7]", n_passages=2) == 0.0


def test_reference_similarity():
    assert reference_similarity("eGFR below 30 and acidosis", "acidosis and eGFR below 30") == 1.0
    assert reference_similarity("weather is mild", "eGFR below 30") == 0.0


def test_relevance_prefers_on_topic_answer():
    j = LexicalJudge()
    q = "Does ibuprofen interact with lithium?"
    on = j.relevance(q, "Yes, ibuprofen can raise serum lithium levels.")
    off = j.relevance(q, "The weather in Washington is mild in September.")
    assert on > 0.5 > off
