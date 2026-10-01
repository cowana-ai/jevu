"""High-level API: give a concept, erase it from text (or from embeddings you already have).

``ConceptScrubber`` owns the whole flow -- it embeds text (if you don't pass embeddings),
labels the concept with JEV (if you don't pass labels), and fits an eraser. The simplest use
is just a concept + text::

    from jevu import ConceptScrubber

    scrubber = ConceptScrubber(concept="Does the text describe a woman?")
    scrubber.fit(texts)                      # embed -> JEV-label -> fit eraser
    clean = scrubber.transform(new_texts)    # embed -> erase  (returns cleaned embeddings)

Bring your own embeddings and/or labels to skip the API calls::

    scrubber.fit(texts=texts, embeddings=X)      # your embeddings, JEV labels the concept
    scrubber.fit(embeddings=X, labels=y)         # your embeddings and labels (no JEV, no OpenAI)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .audit import concept_auc
from .concepts import LLMConceptLabeler
from .embedders import OpenAIEmbedder
from .erasers import InlpEraser, LeaceEraser
from .labelers import JevLabeler

__all__ = ["ConceptScrubber"]

_ERASERS = {"leace": LeaceEraser, "inlp": InlpEraser}


class ConceptScrubber:
    """Erase a target concept from text embeddings.

    Parameters
    ----------
    concept : the concept to erase. A yes/no question when ``expand=False``; a high-level
        attribute (e.g. ``"gender"``) when ``expand=True``.
    method : ``"leace"`` (default, closed-form, minimal-damage) or ``"inlp"`` (iterative).
    expand : if True, an LLM expands ``concept`` into ``n_questions`` sub-questions (woman, man,
        gendered pronouns, ...) and the whole multi-dimensional concept is erased.
    n_questions : number of sub-questions when ``expand=True``.
    labeler : object with ``.score(texts) -> (n,) or (n, k)``; overrides ``concept``/``expand``.
    embedder : object with ``.embed(texts) -> np.ndarray``; defaults to :class:`OpenAIEmbedder`.
    embed_model, jev_model, llm_model : model ids for the embedder / JEV / concept-expansion LLM.
    cache_dir : shared on-disk cache for embeddings, JEV scores, and concept expansions.
    openrouter_api_key, openai_api_key : keys for JEV / (embeddings and the expansion LLM).
    eraser_kwargs : forwarded to the chosen eraser.
    """

    def __init__(self, concept: Optional[str] = None, method: str = "leace", *,
                 expand: bool = False, n_questions: int = 6,
                 labeler=None, embedder=None,
                 embed_model: str = "text-embedding-3-small",
                 jev_model: str = "typesafe/jev-1.13",
                 llm_model: str = "gpt-4o-mini",
                 cache_dir: Optional[str] = None,
                 openrouter_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 **eraser_kwargs):
        if method not in _ERASERS:
            raise ValueError(f"method must be one of {sorted(_ERASERS)}")
        self.concept = concept
        self.method = method
        self.expand = expand
        self.n_questions = n_questions
        self._labeler = labeler
        self._embedder = embedder
        self.embed_model = embed_model
        self.jev_model = jev_model
        self.llm_model = llm_model
        self.cache_dir = cache_dir
        self._openrouter_api_key = openrouter_api_key
        self._openai_api_key = openai_api_key
        self.eraser = _ERASERS[method](**eraser_kwargs)

    # -- lazy components (only built when actually needed) ----------------------
    def _labeler_(self):
        if self._labeler is None:
            if self.concept is None:
                raise ValueError("no labels given and no concept/labeler set; "
                                 "pass concept=..., labeler=..., or labels=... to fit()")
            if self.expand:
                self._labeler = LLMConceptLabeler(
                    self.concept, n_questions=self.n_questions, llm_model=self.llm_model,
                    jev_model=self.jev_model, openai_api_key=self._openai_api_key,
                    openrouter_api_key=self._openrouter_api_key, cache_dir=self.cache_dir)
            else:
                self._labeler = JevLabeler(self.concept, model=self.jev_model,
                                           api_key=self._openrouter_api_key, cache_dir=self.cache_dir)
        return self._labeler

    def _embedder_(self):
        if self._embedder is None:
            self._embedder = OpenAIEmbedder(model=self.embed_model,
                                            api_key=self._openai_api_key, cache_dir=self.cache_dir)
        return self._embedder

    def _resolve_embeddings(self, texts, embeddings) -> np.ndarray:
        if embeddings is not None:
            return np.asarray(embeddings, dtype=float)
        if texts is None:
            raise ValueError("pass texts=... (to embed) or embeddings=...")
        return np.asarray(self._embedder_().embed(texts), dtype=float)

    # -- public API ------------------------------------------------------------
    def fit(self, texts: Optional[Sequence[str]] = None, *, embeddings=None, labels=None) -> "ConceptScrubber":
        X = self._resolve_embeddings(texts, embeddings)
        if labels is None:
            if texts is None:
                raise ValueError("pass labels=... or texts=... so the concept can be scored")
            labeler = self._labeler_()
            labels = labeler.score(texts)
            if hasattr(labeler, "questions"):      # record LLM-expanded sub-questions, if any
                self.concept_questions_ = labeler.questions(texts)
        y = np.asarray(labels, dtype=float)
        if len(y) != len(X):
            raise ValueError("labels/texts and embeddings must have the same length")
        self.eraser.fit(X, y)
        self.labels_ = y
        self._fit_X_ = X
        return self

    def transform(self, texts: Optional[Sequence[str]] = None, *, embeddings=None) -> np.ndarray:
        return self.eraser.transform(self._resolve_embeddings(texts, embeddings))

    def fit_transform(self, texts: Optional[Sequence[str]] = None, *, embeddings=None, labels=None) -> np.ndarray:
        self.fit(texts, embeddings=embeddings, labels=labels)
        return self.eraser.transform(self._fit_X_)

    def audit(self, texts: Optional[Sequence[str]] = None, *, embeddings=None, labels=None) -> dict:
        """Concept recoverability (probe AUC) before vs. after erasure on this data."""
        X = self._resolve_embeddings(texts, embeddings)
        if labels is None:
            if texts is not None:
                labels = self._labeler_().score(texts)
            elif hasattr(self, "labels_") and len(self.labels_) == len(X):
                labels = self.labels_
            else:
                raise ValueError("pass labels=... or texts=... to audit()")
        y = np.asarray(labels, dtype=float)
        Xa = self.eraser.transform(X)

        def mean_auc(Xm):
            if y.ndim == 1:
                return concept_auc(Xm, y)
            return float(np.nanmean([concept_auc(Xm, y[:, j]) for j in range(y.shape[1])]))

        return {
            "concept_auc_before": mean_auc(X),
            "concept_auc_after": mean_auc(Xa),
            "n_concepts": 1 if y.ndim == 1 else int(y.shape[1]),
        }
