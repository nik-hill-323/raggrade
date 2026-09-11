# ragcheck

Evaluation for retrieval-augmented generation. It scores the three places a
RAG answer can go wrong, separately, so you know which part of the pipeline
to fix:

| Failure | Metric | Needs |
|---|---|---|
| Retrieval missed the evidence | `precision_at_k`, `recall_at_k`, `mrr` | ids of relevant passages |
| Generation said something the evidence does not support | `faithfulness` (+ the unsupported sentences) | nothing |
| Answer is grounded but does not answer the question | `answer_relevance` | nothing |
| Answer promises citations and does not deliver | `citation_coverage` | nothing |
| Drift against a known-good answer | `reference_similarity` | a reference answer |

Faithfulness and relevance go through a **judge**. The built-in `LexicalJudge`
is deterministic and offline, so the suite runs in CI on every commit with no
API key. `AnthropicJudge` and `OpenAIJudge` understand paraphrase and negation
and are for real evaluation runs.

```bash
pip install -e ".[dev]"
ragcheck run examples/cases.jsonl
```

```
metric                  mean
precision_at_k          0.833
recall_at_k             0.833
mrr                     1.000
faithfulness            0.556
answer_relevance        0.556
citation_coverage       0.556
reference_similarity    0.667
```

The example set has one clean answer, one answer with an invented final
sentence, and one answer that ignores the question. The per-case output names
the invented sentence:

```python
from ragcheck import evaluate
from ragcheck.runner import load_cases

summary = evaluate(load_cases("examples/cases.jsonl"))
summary.results[1].unsupported_claims
# ['Sertraline cures depression in two weeks for all patients.']
```

## Gate a deployment on it

`ragcheck gate` exits non-zero when a results file misses a threshold, so a
regression in faithfulness fails the build:

```bash
ragcheck run eval/cases.jsonl --judge anthropic --out results.json
ragcheck gate results.json --faithfulness 0.9 --recall 0.8 --citations 0.95
```

```
PASS  faithfulness       0.940 >= 0.9
FAIL  recall_at_k        0.740 >= 0.8
PASS  citation_coverage  1.000 >= 0.95
```

## Case format

One JSON object per line. Only `id`, `question` and `answer` are required.

```json
{"id": "q1",
 "question": "When is metformin contraindicated?",
 "answer": "Metformin is contraindicated below eGFR 30. [1]",
 "retrieved": [{"id": "label-metformin", "text": "Metformin is contraindicated in severe renal impairment (eGFR below 30) ..."}],
 "relevant_ids": ["label-metformin"],
 "reference_answer": "Severe renal impairment and metabolic acidosis."}
```

Citation markers are `[n]`, 1-indexed into `retrieved`, the convention used
by [medrag-toolkit](https://github.com/nik-hill-323/medrag-toolkit), which
this was built to evaluate.

## How faithfulness is computed

The answer is split into sentences, citation markers are stripped, and the
judge is asked whether each sentence is supported by at least one retrieved
passage. Faithfulness is the supported fraction. This is the same
decomposition RAGAS uses; the difference is that the judge is a small
interface (`supports(claim, passages)` and `relevance(question, answer)`), so
you can swap in a cheap offline judge for CI and an LLM for accuracy, or write
your own.

## Tests

```bash
pytest
```
