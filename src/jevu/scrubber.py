"""High-level API: give a concept and your embeddings, erase the concept.

Bring embeddings from **any** model — this package never embeds for you. Give it the embeddings plus
either the raw texts (so JEV can label the concept) or your own labels::

    from jevu import ConceptScrubber

    scrubber = ConceptScrubber(concept="Does the text describe a woman?")
    scrubber.fit(X, texts=texts)         # your embeddings; JEV labels the concept
    X_clean = scrubber.transform(X_new)  # erase the concept from new embeddings

    scrubber.fit(X, labels=y)            # your embeddings and labels (no JEV at all)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .audit import concept_auc
from .concepts import LLMConceptLabeler
from .erasers import InlpEraser, LeaceEraser
from .labelers import JevLabeler

__all__ = ["ConceptScrubber"]

_ERASERS = {"leace": LeaceEraser, "inlp": InlpEraser}


class ConceptScrubber:
    """Erase a target concept from embeddings you provide.

    Parameters
    ----------
    concept : the concept to erase. A yes/no question when ``expand=False``; a high-level
        attribute (e.g. ``"gender"``) when ``expand=True``.
    method : ``"leace"`` (default, closed-form, minimal-damage) or ``"inlp"`` (iterative).
    expand : if True, an LLM expands ``concept`` into ``n_questions`` sub-questions and the whole
        multi-dimensional concept is erased at once.
    n_questions : number of sub-questions when ``expand=True``.
    labeler : object with ``.score(texts) -> (n,) or (n, k)``; overrides ``concept``/``expand``.
    jev_model, llm_model : model ids for JEV scoring / concept expansion.
    cache_dir : shared on-disk cache for JEV scores and concept expansions.
    openrouter_api_key, openai_api_key : keys for JEV / the expansion LLM (else read from env).
    eraser_kwargs : forwarded to the chosen eraser.
    """

    def __init__(self, concept: Optional[str] = None, method: str = "leace", *,
                 expand: bool = False, n_questions: int = 6, labeler=None,
                 jev_model: str = "typesafe/jev-1.13", llm_model: str = "gpt-4o-mini",
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
        self.jev_model = jev_model
        self.llm_model = llm_model
        self.cache_dir = cache_dir
        self._openrouter_api_key = openrouter_api_key
        self._openai_api_key = openai_api_key
        self.eraser = _ERASERS[method](**eraser_kwargs)

    def _labeler_(self):
        if self._labeler is None:
            if self.concept is None:
                raise ValueError("no labels given and no concept/labeler set; "
                                 "pass concept=..., labeler=..., or labels=...")
            if self.expand:
                self._labeler = LLMConceptLabeler(
                    self.concept, n_questions=self.n_questions, llm_model=self.llm_model,
                    jev_model=self.jev_model, openai_api_key=self._openai_api_key,
                    openrouter_api_key=self._openrouter_api_key, cache_dir=self.cache_dir)
            else:
                self._labeler = JevLabeler(self.concept, model=self.jev_model,
                                           api_key=self._openrouter_api_key, cache_dir=self.cache_dir)
        return self._labeler

    def _labels_for(self, X, texts, labels) -> np.ndarray:
        if labels is None:
            if texts is None:
                raise ValueError("pass labels=... or texts=... so the concept can be scored")
            labeler = self._labeler_()
            labels = labeler.score(texts)
            if hasattr(labeler, "questions"):           # record LLM-expanded sub-questions, if any
                self.concept_questions_ = labeler.questions(texts)
        y = np.asarray(labels, dtype=float)
        if len(y) != len(X):
            raise ValueError("labels/texts and embeddings must have the same length")
        return y

    # -- public API ------------------------------------------------------------
    def fit(self, embeddings, *, texts: Optional[Sequence[str]] = None, labels=None) -> "ConceptScrubber":
        X = np.asarray(embeddings, dtype=float)
        y = self._labels_for(X, texts, labels)
        self.eraser.fit(X, y)
        self.labels_ = y
        return self

    def transform(self, embeddings) -> np.ndarray:
        return self.eraser.transform(np.asarray(embeddings, dtype=float))

    def fit_transform(self, embeddings, *, texts: Optional[Sequence[str]] = None, labels=None) -> np.ndarray:
        self.fit(embeddings, texts=texts, labels=labels)
        return self.transform(embeddings)

    def concept_scores(self, texts: Sequence[str]) -> np.ndarray:
        """Calibrated concept score(s) per text via the labeler, without fitting an eraser.

        Shape ``(n,)`` for a single concept, or ``(n, k)`` when the concept is LLM-expanded.
        """
        return np.asarray(self._labeler_().score(list(texts)), dtype=float)

    def audit(self, embeddings, *, texts: Optional[Sequence[str]] = None, labels=None) -> dict:
        """Concept recoverability (probe AUC) before vs. after erasure on these embeddings."""
        X = np.asarray(embeddings, dtype=float)
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
