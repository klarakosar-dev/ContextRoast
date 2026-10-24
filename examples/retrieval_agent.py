"""A user-defined local FAQ retriever, independent of ContextRoast's mock provider.

Every context field is searched. No field names, answers, or task dependencies are
hardcoded. This example uses no network; arbitrary entrypoints may incur charges.
"""

import re
from collections import Counter
from math import sqrt

from contextroast.python_provider import AgentCase

_STOP_WORDS = {
    "a",
    "an",
    "the",
    "i",
    "my",
    "we",
    "our",
    "how",
    "do",
    "can",
    "is",
    "to",
    "for",
    "of",
    "and",
    "where",
    "what",
}


def _tokens(text: str) -> Counter:
    return Counter(
        word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in _STOP_WORDS
    )


def predict(case: AgentCase, context: dict) -> dict:
    """Find the closest FAQ question by cosine similarity over word counts."""
    query = _tokens(case.input)
    best_score, best = 0.0, None
    for records in context.values():
        for record in records:
            document = _tokens(record["question"])
            overlap = sum(count * document[word] for word, count in query.items())
            norm = sqrt(
                sum(count * count for count in query.values())
                * sum(count * count for count in document.values())
            )
            score = overlap / norm if norm else 0.0
            if score > best_score:
                best_score, best = score, record
    if best is None:
        return {"article_id": None, "answer": None}
    return {"article_id": best["id"], "answer": best["answer"]}
