"""LLM-expanded concept labeling.

Give a high-level attribute to erase (e.g. ``"gender"``); an LLM turns it into several concrete
yes/no questions covering its facets (woman / man / gendered pronouns / ...), and laya scores each
one per text locally. The result is an ``(n_texts, n_questions)`` matrix -- a multi-dimensional
concept that the erasers remove as a whole.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .laya_labeler import LayaLabeler

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
    "- MANY-VALUED IDENTITY (occupation, nationality, topic, product category): the attribute takes "
    "MANY distinct values, so a few BROAD umbrella questions will NOT remove it -- each erased question "
    "removes roughly one direction, and an umbrella like 'a healthcare professional' collapses many "
    "distinct values (nurse, physician, dentist, ...) onto a single direction. Instead enumerate "
    "FINE-GRAINED, SPECIFIC values so the set SPANS the full range seen in the examples: aim for about "
    "one question per distinct value (up to the cap), each detecting a SINGLE specific value (e.g. for "
    "occupation: refers to a nurse; refers to a physician; refers to a software engineer; refers to an "
    "attorney; refers to a teacher; refers to a journalist; refers to an accountant; ...). "
    "The set must be:\n"
    "  (a) DIVERSE and minimally redundant -- no two questions near-synonyms, and never nest one inside "
    "another (do NOT include both 'a healthcare professional' and 'a nurse'). Each question should add a "
    "NEW direction: knowing the answers to the others should not let you predict it.\n"
    "  (b) COMPLETE on the rare values too -- give the minority/less-common values their OWN specific "
    "questions, not just the frequent ones; a value with no question of its own cannot be removed.\n"
    "  (c) RECONSTRUCTIVE -- from the full set of answers one should be able to tell exactly WHICH value "
    "applies. Prefer specific over broad; granularity and independence matter more than tidy grouping.\n"
    "CONCRETE EXAMPLE for the attribute 'occupation'.\n"
    "  BAD (too broad -- a few umbrellas that collapse dozens of jobs onto a few directions; do NOT do "
    "this): 'Does the text describe a healthcare professional?'; 'Does the text describe an educator or "
    "academic?'; 'Does the text describe an artist or performer?'\n"
    "  GOOD (specific, one job each, spanning the range): 'Does the text describe a nurse?'; 'Does the "
    "text describe a physician?'; 'Does the text describe a surgeon?'; 'Does the text describe a "
    "dentist?'; 'Does the text describe a professor?'; 'Does the text describe a teacher?'; 'Does the "
    "text describe a software engineer?'; 'Does the text describe an attorney?'; 'Does the text describe "
    "a journalist?'; 'Does the text describe a photographer?'; 'Does the text describe an accountant?'; "
    "'Does the text describe a psychologist?'; ...\n"
    "Mirror the GOOD style: name a SPECIFIC value in each question.\n"
    "Every question: a yes/no question starting like 'Does the text ...?'; answerable from the text "
    "alone; role-neutral (query or document); no two near-duplicates. Avoid questions that are yes -- "
    "or no -- for essentially EVERY text (they carry no information); a specific value question that is "
    "yes for only a small minority is exactly what you want.\n"
    "Return a JSON object with exactly one key, 'questions', containing exactly %d distinct question "
    "strings."
)

AUTO_EXPANSION_SYSTEM = (
    "You convert an attribute that should be removed from text embeddings into the SMALLEST set of "
    "yes/no DETECTION questions that reliably detects it. First judge the attribute's cardinality from "
    "the attribute itself and the example texts:\n"
    "- BINARY / low-dimensional (gender, sentiment, formality, 'is it a question'): ONE well-phrased "
    "question is enough (use two only if two clearly distinct sides exist).\n"
    "- MANY-VALUED IDENTITY (occupation, nationality, topic, product category): you need MANY "
    "fine-grained questions -- roughly one per DISTINCT value seen in the examples (not broad umbrellas, "
    "which collapse several values onto one direction and fail to remove the attribute). Make them "
    "diverse and minimally redundant (never nest one value inside another), and give the rare values "
    "their own questions too.\n"
    "Use enough questions to SPAN the attribute's values -- for a many-valued identity that means close "
    "to its cardinality -- and never more than %d.\n"
    "Every question: a yes/no question starting like 'Does the text ...?'; answerable from the text "
    "alone; role-neutral; no near-duplicates; and never yes (or no) for essentially every text.\n"
    "Return a JSON object with keys 'cardinality' ('binary' or 'identity') and 'questions' (the list)."
)


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class LLMConceptLabeler:
    """Expand a concept with an LLM, then label each sub-question locally with laya.

    The concept may be terse or ambiguous (``"genders"``, ``"age"``, ``"tone"``) -- the LLM
    interprets it and writes concrete, discriminative detection questions, so you don't have to
    phrase a clean yes/no question yourself.

    Parameters
    ----------
    concept : attribute to erase; terse/ambiguous input is fine (e.g. ``"gender"``, ``"sentiment"``).
    n_questions : how many yes/no questions to generate.
    questions : provide questions directly to skip the LLM call entirely.
    llm_model : chat model used to expand the concept.
    laya_model : laya checkpoint used to score each question locally.
    openai_api_key : key for the expansion LLM (else read from ``$OPENAI_API_KEY``).
    device : torch device for laya (``None`` = auto, e.g. ``mps``/``cuda``/``cpu``).
    cache_dir : shared cache for the expansion and the laya scores.
    llm_client / laya_agent : optional injected clients (for tests / reuse).
    """

    def __init__(self, concept: str, *, n_questions=6, max_questions: int = 24,
                 questions: Optional[Sequence[str]] = None,
                 llm_model: str = "gpt-4o-mini", laya_model: str = "convaiinnovations/laya",
                 openai_api_key: Optional[str] = None, device: Optional[str] = None,
                 cache_dir: Optional[str] = None, llm_client=None, laya_agent=None,
                 batch_size: int = 16, progress: bool = True, temperature: float = 0.0):
        if not concept or not concept.strip():
            raise ValueError("concept must be a non-empty string")
        if n_questions != "auto" and not (isinstance(n_questions, int) and n_questions >= 1):
            raise ValueError("n_questions must be a positive int or 'auto'")
        self.concept = concept.strip()
        self.temperature = temperature          # 0 = deterministic & reproducible expansion
        self.n_questions = n_questions          # int (fixed count) or 'auto' (LLM decides)
        self.max_questions = max_questions      # cap when n_questions == 'auto'
        self._questions = [q.strip() for q in questions] if questions else None
        self.llm_model = llm_model
        self.laya_model = laya_model
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.llm_client = llm_client
        self.laya_agent = laya_agent
        self.batch_size = batch_size
        self.progress = progress
        self._scorer = None

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
        request = {"model": self.llm_model, "messages": messages, "temperature": self.temperature,
                   "response_format": {"type": "json_object"}}
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

    def _scorer_(self) -> LayaLabeler:
        """A shared laya scorer (one loaded model reused across calls)."""
        if self._scorer is None:
            self._scorer = LayaLabeler(
                questions=self._questions or [self.concept], model=self.laya_model,
                device=self.device, batch_size=self.batch_size,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                agent=self.laya_agent, progress=self.progress)
            self.laya_agent = self._scorer._agent  # reuse loaded model
        return self._scorer

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n_texts, n_questions)`` matrix of laya scores for the concept's questions."""
        return self.score_questions(self.questions(texts), texts)

    def score_questions(self, questions: Sequence[str], texts: Sequence[str]) -> np.ndarray:
        """Score an explicit list of questions over texts -> ``(n_texts, len(questions))`` with laya.

        laya answers all questions for a text in one local forward pass; scores are cached per
        ``(model, question, text)`` so re-runs and repeated texts are free.
        """
        scorer = self._scorer_()
        M = scorer.score_questions(list(questions), list(texts))
        self.laya_agent = scorer._agent  # keep the loaded model for subsequent calls
        return M
