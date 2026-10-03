"""LLM-expanded concept labeling.

Give a high-level attribute to erase (e.g. ``"gender"``); an LLM turns it into several concrete
yes/no questions covering its facets (woman / man / gendered pronouns / ...), and JEV scores each
one per text. The result is an ``(n_texts, n_questions)`` matrix -- a multi-dimensional concept
that the erasers remove as a whole.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ._util import progress
from .labelers import JevLabeler

logger = logging.getLogger(__name__)

__all__ = ["LLMConceptLabeler"]

EXPANSION_SYSTEM = (
    "You convert an attribute that should be removed from text embeddings into concrete yes/no "
    "DETECTION questions.\n"
    "The attribute may be given tersely or ambiguously -- a single word, a plural, a typo, or a short "
    "phrase (e.g. 'genders', 'age', 'occupation', 'tone'). Do NOT score the raw phrase; interpret it "
    "charitably, picking the most common sensible reading if ambiguous. Use the provided example texts "
    "to see how the attribute actually varies in this data.\n"
    "Choose the question style by the attribute's cardinality:\n"
    "- BINARY / low-dimensional (gender, sentiment, formality): write questions that each SPLIT the "
    "texts fairly evenly, covering the attribute's sides and facets (e.g. for gender: refers to a "
    "woman; refers to a man; uses gendered pronouns/titles).\n"
    "- MANY-VALUED IDENTITY (occupation, nationality, topic, product category): enumerate DISTINCT "
    "SUB-CATEGORIES that TOGETHER COVER the values seen in the examples; each question detects one "
    "sub-category and may be true for only a minority (e.g. for occupation: is a healthcare "
    "professional; is an educator or academic; is an engineer or technical worker; is an artist or "
    "performer; is a legal professional; ...). The set, taken together, should let someone reconstruct "
    "the attribute -- so prefer broad coverage of the categories present over any single question.\n"
    "Every question: a yes/no question starting like 'Does the text ...?'; answerable from the text "
    "alone; role-neutral (query or document); no two near-duplicates. Avoid questions that are yes -- "
    "or no -- for essentially EVERY text (they carry no information); minority-yes sub-category "
    "questions are fine when the set covers the range.\n"
    "Return a JSON object with exactly one key, 'questions', containing exactly %d distinct question "
    "strings."
)

AUTO_EXPANSION_SYSTEM = (
    "You convert an attribute that should be removed from text embeddings into the SMALLEST set of "
    "yes/no DETECTION questions that reliably detects it. First judge the attribute's cardinality from "
    "the attribute itself and the example texts:\n"
    "- BINARY / low-dimensional (gender, sentiment, formality, 'is it a question'): ONE well-phrased "
    "question is enough (use two only if two clearly distinct sides exist).\n"
    "- MANY-VALUED IDENTITY (occupation, nationality, topic, product category): you need several "
    "covering sub-category questions -- enumerate distinct categories that TOGETHER COVER the values "
    "seen in the examples (each may be true for only a minority).\n"
    "Use the FEWEST questions that still detect the attribute well, and never more than %d.\n"
    "Every question: a yes/no question starting like 'Does the text ...?'; answerable from the text "
    "alone; role-neutral; no near-duplicates; and never yes (or no) for essentially every text.\n"
    "Return a JSON object with keys 'cardinality' ('binary' or 'identity') and 'questions' (the list)."
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

    def __init__(self, concept: str, *, n_questions=6, max_questions: int = 24,
                 questions: Optional[Sequence[str]] = None,
                 llm_model: str = "gpt-4o-mini", jev_model: str = "typesafe/jev-1.13",
                 openai_api_key: Optional[str] = None, openrouter_api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None, llm_client=None, jev_client=None,
                 max_workers: int = 8, progress: bool = True):
        if not concept or not concept.strip():
            raise ValueError("concept must be a non-empty string")
        if n_questions != "auto" and not (isinstance(n_questions, int) and n_questions >= 1):
            raise ValueError("n_questions must be a positive int or 'auto'")
        self.concept = concept.strip()
        self.n_questions = n_questions          # int (fixed count) or 'auto' (LLM decides)
        self.max_questions = max_questions      # cap when n_questions == 'auto'
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
        self.progress = progress

    def _generate(self, texts: Optional[Sequence[str]]) -> list:
        auto = self.n_questions == "auto"
        evidence = {"attribute_to_erase": self.concept}
        if texts:
            evidence["example_texts"] = list(texts)[:20]
        system = AUTO_EXPANSION_SYSTEM % self.max_questions if auto else EXPANSION_SYSTEM % self.n_questions
        messages = [
            {"role": "system", "content": system},
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
        body = json.loads(resp.choices[0].message.content)
        qs = [q.strip() for q in body["questions"] if isinstance(q, str) and q.strip()]
        qs = list(dict.fromkeys(qs))
        qs = qs[: self.max_questions] if auto else qs[: self.n_questions]
        if not qs:
            raise ValueError("LLM returned no usable questions")
        if auto:
            logger.info("auto-expand judged %r as %s -> %d question(s)",
                        self.concept, body.get("cardinality", "?"), len(qs))
        if path:
            path.write_text(json.dumps({"concept": self.concept, "cardinality": body.get("cardinality"),
                                        "questions": qs}, ensure_ascii=False))
        return qs

    def questions(self, texts: Optional[Sequence[str]] = None) -> list:
        """The yes/no questions this concept expands to (generated once, then cached)."""
        if self._questions is None:
            self._questions = self._generate(texts)
            logger.info("expanded concept %r into %d questions: %s",
                        self.concept, len(self._questions), self._questions)
        return self._questions

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n_texts, n_questions)`` matrix of JEV scores for the concept's questions."""
        return self.score_questions(self.questions(texts), texts)

    def score_questions(self, questions: Sequence[str], texts: Sequence[str]) -> np.ndarray:
        """Score an explicit list of questions over texts -> ``(n_texts, len(questions))``.

        All ``questions x texts`` cells are scored in one shared thread pool (one pooled HTTP client),
        so the whole job runs at ``max_workers`` concurrency rather than one question at a time.
        """
        texts = list(texts)
        qs = list(questions)
        labs = [JevLabeler(q, model=self.jev_model, api_key=self.openrouter_api_key,
                           cache_dir=str(self.cache_dir) if self.cache_dir else None,
                           max_workers=self.max_workers) for q in qs]
        tasks = [(ti, ci, labs[ci], t) for ci in range(len(qs)) for ti, t in enumerate(texts)]
        n_missing = sum(1 for _, _, lab, t in tasks
                        if not (lab.cache_dir and lab._cache_path(lab._payload(t)).exists()))
        logger.info("scoring %d texts x %d questions = %d cells (%d cached, %d to fetch)",
                    len(texts), len(qs), len(tasks), len(tasks) - n_missing, n_missing)
        client = (self.jev_client or labs[0]._make_client()) if n_missing else None
        close = n_missing and self.jev_client is None
        M = np.empty((len(texts), len(qs)), dtype=float)
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futs = {pool.submit(lab._score_one, t, client): (ti, ci)
                        for ti, ci, lab, t in tasks}
                desc = f"JEV x{len(qs)}: {self.concept[:30]}"
                for fut in progress(as_completed(futs), total=len(tasks), desc=desc,
                                    enabled=self.progress and len(tasks) > 1):
                    ti, ci = futs[fut]
                    M[ti, ci] = fut.result()
            return M
        finally:
            if close and client is not None and hasattr(client, "close"):
                client.close()
