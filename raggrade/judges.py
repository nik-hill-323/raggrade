"""Judges decide whether a claim is supported by a passage and whether an
answer addresses a question.

``LexicalJudge`` is deterministic and offline: it uses content-word overlap
and TF-IDF cosine similarity, which is enough to catch answers that wander
away from their sources and to run in CI on every commit. The LLM judges are
stricter and understand paraphrase and negation; use them for real
evaluation runs.
"""

from __future__ import annotations

import json
import os
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text import overlap, tokens


class Judge(Protocol):
    name: str

    def supports(self, claim: str, passages: list[str]) -> bool:
        """Is the claim entailed by at least one of the passages?"""

    def relevance(self, question: str, answer: str) -> float:
        """How well does the answer address the question? 0 to 1."""


class LexicalJudge:
    """Offline judge. A claim is supported when at least ``threshold`` of its
    content words appear in some single retrieved passage."""

    name = "lexical"

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def supports(self, claim: str, passages: list[str]) -> bool:
        if not tokens(claim):
            return True
        return any(overlap(claim, p) >= self.threshold for p in passages)

    def relevance(self, question: str, answer: str) -> float:
        if not tokens(question) or not tokens(answer):
            return 0.0
        vec = TfidfVectorizer(tokenizer=tokens, token_pattern=None, lowercase=False)
        m = vec.fit_transform([question, answer])
        sim = float(cosine_similarity(m[0], m[1])[0, 0])
        # Cosine between a short question and a longer answer is small even when
        # the answer is on topic; also credit direct coverage of the question's terms.
        return max(sim, overlap(question, answer))


_SUPPORT_PROMPT = """You are checking whether a claim is supported by source passages.

Passages:
{passages}

Claim: {claim}

Answer with a JSON object {{"supported": true}} if the passages state or directly
imply the claim, otherwise {{"supported": false}}. A claim that goes beyond what the
passages say, or contradicts them, is not supported."""

_RELEVANCE_PROMPT = """Rate how well the answer addresses the question, ignoring whether it is correct.

Question: {question}
Answer: {answer}

Reply with a JSON object {{"score": s}} where s is 0 (does not address the question),
0.5 (partially addresses it), or 1 (fully addresses it)."""


def _parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


class AnthropicJudge:
    """Uses the Anthropic API. Requires ``pip install raggrade[anthropic]`` and
    ``ANTHROPIC_API_KEY``."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def _ask(self, prompt: str) -> dict:
        msg = self.client.messages.create(
            model=self.model, max_tokens=64, messages=[{"role": "user", "content": prompt}]
        )
        return _parse_json(msg.content[0].text)

    def supports(self, claim: str, passages: list[str]) -> bool:
        joined = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        return bool(self._ask(_SUPPORT_PROMPT.format(passages=joined, claim=claim)).get("supported"))

    def relevance(self, question: str, answer: str) -> float:
        return float(self._ask(_RELEVANCE_PROMPT.format(question=question, answer=answer)).get("score", 0))


class OpenAIJudge:
    """Uses the OpenAI API. Requires ``pip install raggrade[openai]`` and ``OPENAI_API_KEY``."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        import openai

        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def _ask(self, prompt: str) -> dict:
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=64, messages=[{"role": "user", "content": prompt}]
        )
        return _parse_json(r.choices[0].message.content or "")

    def supports(self, claim: str, passages: list[str]) -> bool:
        joined = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        return bool(self._ask(_SUPPORT_PROMPT.format(passages=joined, claim=claim)).get("supported"))

    def relevance(self, question: str, answer: str) -> float:
        return float(self._ask(_RELEVANCE_PROMPT.format(question=question, answer=answer)).get("score", 0))


def get_judge(name: str, **kwargs) -> Judge:
    if name == "lexical":
        return LexicalJudge(**kwargs)
    if name == "anthropic":
        return AnthropicJudge(**kwargs)
    if name == "openai":
        return OpenAIJudge(**kwargs)
    raise ValueError(f"unknown judge {name!r}; choose lexical, anthropic or openai")
