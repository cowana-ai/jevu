"""LLM-expanded concept labeling.

Give a high-level attribute to erase (e.g. ``"gender"``); an LLM turns it into several concrete
yes/no questions covering its facets (woman / man / gendered pronouns / ...), and JEV scores each
one per text. The result is an ``(n_texts, n_questions)`` matrix -- a multi-dimensional concept
that the erasers remove as a whole.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .labelers import JevLabeler

__all__ = ["LLMConceptLabeler"]

EXPANSION_SYSTEM = (
    "You convert an attribute that should be removed from text embeddings into concrete yes/no "
    "DETECTION questions.\n"
    "The attribute may be given tersely or ambiguously -- a single word, a plural, a typo, a short "
    "phrase, or an underspecified topic (e.g. 'genders', 'age', 'tone', 'politics'). Do NOT score "
    "the raw phrase; first interpret it charitably as a well-defined attribute of a person or text, "
    "picking the most common sensible reading if it is ambiguous, then enumerate its main facets.\n"
    "Write questions that DETECT that attribute, each scored independently on ONE text. Requirements "
    "for EVERY question: it is a yes/no question starting like 'Does the text ...?'; it is answerable "
    "from the text alone; it is role-neutral (works for a short query or a long document); it is "
    "concrete and DISCRIMINATIVE so it splits texts roughly -- never almost-always-yes or "
    "almost-always-no; and no two questions are near-duplicates.\n"
    "Spread the questions across the attribute's distinct facets. For example, for 'gender': refers "
    "to a woman or female person; refers to a man or male person; uses gendered pronouns, titles, or "
    "names. If the attribute is naturally binary, still phrase each side as its own detection "
    "question rather than one vague question.\n"
    "Return a JSON object with exactly one key, 'questions', containing exactly %d distinct question "
    "strings."
)


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class LLMConceptLabeler:
    """Expand a concept with an LLM, then label each sub-question with JEV.

    The concept may be terse or ambiguous (``"genders"``, ``"age"``, ``"tone"``) -- the LLM
    interprets it and writes concrete, discriminative detection questions, so you don't have to
    phrase a clean yes/no question yourself.

    Parameters
    ----------
    concept : attribute to erase; terse/ambiguous input is fine (e.g. ``"gender"``, ``"sentiment"``).
    n_questions : how many yes/no questions to generate.
    questions : provide questions directly to skip the LLM call entirely.
    llm_model : chat model used to expand the concept.
    jev_model : JEV model used to score each question.
    openai_api_key / openrouter_api_key : keys for the LLM / JEV (else read from env).
    cache_dir : shared cache for the expansion and the JEV scores.
    llm_client / jev_client : optional injected clients (for tests / custom endpoints).
    """

    def __init__(self, concept: str, *, n_questions: int = 6,
                 questions: Optional[Sequence[str]] = None,
                 llm_model: str = "gpt-4o-mini", jev_model: str = "typesafe/jev-1.13",
                 openai_api_key: Optional[str] = None, openrouter_api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None, llm_client=None, jev_client=None,
                 max_workers: int = 8):
        if not concept or not concept.strip():
            raise ValueError("concept must be a non-empty string")
        self.concept = concept.strip()
        self.n_questions = n_questions
        self._questions = [q.strip() for q in questions] if questions else None
        self.llm_model = llm_model
        self.jev_model = jev_model
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.llm_client = llm_client
        self.jev_client = jev_client
        self.max_workers = max_workers

    def _generate(self, texts: Optional[Sequence[str]]) -> list:
        evidence = {"attribute_to_erase": self.concept}
        if texts:
            evidence["example_texts"] = list(texts)[:20]
        messages = [
            {"role": "system", "content": EXPANSION_SYSTEM % self.n_questions},
            {"role": "user", "content": "Generate the questions.\n" + json.dumps(evidence, ensure_ascii=False)},
        ]
        request = {"model": self.llm_model, "messages": messages, "response_format": {"type": "json_object"}}
        path = self.cache_dir / f"expand_{_digest(request)}.json" if self.cache_dir else None
        if path and path.exists():
            return json.loads(path.read_text())["questions"]
        if self.llm_client is None and not self.openai_api_key:
            raise ValueError("no cached expansion and no OpenAI key/client to generate questions")
        client = self.llm_client
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key, timeout=90)
        resp = client.chat.completions.create(**request)
        qs = json.loads(resp.choices[0].message.content)["questions"]
        qs = [q.strip() for q in qs if isinstance(q, str) and q.strip()]
        qs = list(dict.fromkeys(qs))[: self.n_questions]
        if not qs:
            raise ValueError("LLM returned no usable questions")
        if path:
            path.write_text(json.dumps({"concept": self.concept, "questions": qs}, ensure_ascii=False))
        return qs

    def questions(self, texts: Optional[Sequence[str]] = None) -> list:
        """The yes/no questions this concept expands to (generated once, then cached)."""
        if self._questions is None:
            self._questions = self._generate(texts)
        return self._questions

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n_texts, n_questions)`` matrix of JEV scores, one column per sub-question."""
        texts = list(texts)
        cols = []
        for q in self.questions(texts):
            lab = JevLabeler(q, model=self.jev_model, api_key=self.openrouter_api_key,
                             cache_dir=str(self.cache_dir) if self.cache_dir else None,
                             client=self.jev_client, max_workers=self.max_workers)
            cols.append(lab.score(texts))
        return np.column_stack(cols)
