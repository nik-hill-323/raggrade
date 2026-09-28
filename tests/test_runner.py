import json
from pathlib import Path

import pytest

from raggrade import evaluate
from raggrade.cli import main
from raggrade.runner import load_cases

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "cases.jsonl"


def test_example_set_scores_as_expected():
    summary = evaluate(load_cases(EXAMPLES))
    by_id = {r.id: r for r in summary.results}

    # q1: fully grounded, cited, retrieval hit at rank 1
    assert by_id["q1"].faithfulness == 1.0
    assert by_id["q1"].citation_coverage == 1.0
    assert by_id["q1"].mrr == 1.0

    # q2: one invented sentence out of three, and it is uncited
    assert by_id["q2"].faithfulness == pytest.approx(2 / 3)
    assert by_id["q2"].unsupported_claims == ["Sertraline cures depression in two weeks for all patients."]
    assert by_id["q2"].citation_coverage == pytest.approx(2 / 3)

    # q3: off-topic answer, partial recall
    assert by_id["q3"].faithfulness == 0.0
    assert by_id["q3"].answer_relevance < 0.2
    assert by_id["q3"].recall_at_k == 0.5

    assert summary.n == 3
    assert summary.judge == "lexical"
    assert set(summary.means) >= {"faithfulness", "answer_relevance", "citation_coverage", "recall_at_k"}


def test_cli_run_and_gate(tmp_path, capsys):
    out = tmp_path / "results.json"
    assert main(["run", str(EXAMPLES), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["n"] == 3

    assert main(["gate", str(out), "--citations", "0.5"]) == 0
    assert main(["gate", str(out), "--faithfulness", "0.99"]) == 1
    text = capsys.readouterr().out
    assert "FAIL  faithfulness" in text


def test_load_json_list(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"id": "x", "question": "q?", "answer": "a."}]))
    cases = load_cases(p)
    assert cases[0].id == "x" and cases[0].retrieved == []
