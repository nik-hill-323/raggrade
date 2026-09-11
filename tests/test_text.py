from ragcheck.text import citations, overlap, sentences, strip_citations, tokens


def test_sentences_keep_citations_attached():
    s = sentences("Metformin is contraindicated below eGFR 30. [1] It is first line. [2]")
    assert s == ["Metformin is contraindicated below eGFR 30. [1]", "It is first line. [2]"]


def test_citations_and_strip():
    assert citations("Dose is 50 mg. [1][3]") == [1, 3]
    assert strip_citations("Dose is 50 mg. [1][3]") == "Dose is 50 mg."


def test_tokens_drop_stopwords():
    assert tokens("The dose of the drug is 50 mg") == ["dose", "drug", "50", "mg"]


def test_overlap_is_directional():
    assert overlap("renal impairment", "severe renal impairment and acidosis") == 1.0
    assert overlap("severe renal impairment and acidosis", "renal impairment") == 0.5
