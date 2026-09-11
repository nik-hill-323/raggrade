"""    ragcheck run cases.jsonl [--judge lexical|anthropic|openai] [--out results.json]
    ragcheck gate results.json --faithfulness 0.9 --recall 0.8
"""

from __future__ import annotations

import argparse
import json
import sys

from .judges import get_judge
from .runner import evaluate, load_cases


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragcheck", description="Evaluate a RAG system")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="score a file of cases")
    run.add_argument("cases", help="JSON list or JSONL of EvalCase objects")
    run.add_argument("--judge", default="lexical", choices=["lexical", "anthropic", "openai"])
    run.add_argument("--model", default=None, help="model name for LLM judges")
    run.add_argument("--out", default=None, help="write full results as JSON here")

    gate = sub.add_parser("gate", help="exit non-zero if a results file misses thresholds")
    gate.add_argument("results")
    gate.add_argument("--faithfulness", type=float, default=None)
    gate.add_argument("--relevance", type=float, default=None)
    gate.add_argument("--recall", type=float, default=None)
    gate.add_argument("--citations", type=float, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        kwargs = {"model": args.model} if args.model else {}
        summary = evaluate(load_cases(args.cases), get_judge(args.judge, **kwargs))
        print(summary.table())
        if args.out:
            with open(args.out, "w") as f:
                f.write(summary.model_dump_json(indent=2))
        return 0

    with open(args.results) as f:
        means = json.load(f)["means"]
    checks = {
        "faithfulness": args.faithfulness,
        "answer_relevance": args.relevance,
        "recall_at_k": args.recall,
        "citation_coverage": args.citations,
    }
    failed = False
    for metric, threshold in checks.items():
        if threshold is None:
            continue
        value = means.get(metric)
        status = "PASS" if value is not None and value >= threshold else "FAIL"
        failed |= status == "FAIL"
        shown = "n/a" if value is None else f"{value:.3f}"
        print(f"{status}  {metric:<18} {shown} >= {threshold}")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
