"""High-level API: label a concept (via JEV or your own labels) and erase it."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .audit import concept_auc
from .erasers import InlpEraser, LeaceEraser
from .labelers import JevLabeler

__all__ = ["ConceptScrubber"]

_ERASERS = {"leace": LeaceEraser, "inlp": InlpEraser}


class ConceptScrubber:
    """Erase a target concept from embeddings.

    Typical use::

        scrubber = ConceptScrubber(concept="Does the text describe a woman?", method="leace")
        scrubber.fit(embeddings=X, texts=texts)      # JEV labels the concept, fits the eraser
        X_clean = scrubber.transform(X_new)          # scrub unseen embeddings

    If you already have concept labels, skip JEV entirely::

        scrubber.fit(embeddings=X, labels=y)

    Parameters
    ----------
    concept : concept as a yes/no question (used to build a :class:`JevLabeler` if none given).
    method : ``"leace"`` (default, closed-form, minimal-damage) or ``"inlp"`` (iterative).
    labeler : a pre-built labeler; overrides ``concept``. Any object with ``.score(texts)``.
    eraser_kwargs : forwarded to the chosen eraser.
    """

    def __init__(self, concept: Optional[str] = None, method: str = "leace",
                 labeler=None, **eraser_kwargs):
        if method not in _ERASERS:
            raise ValueError(f"method must be one of {sorted(_ERASERS)}")
        if labeler is None and concept is None:
            raise ValueError("provide either a concept, a labeler, or pass labels to fit()")
        self.concept = concept
        self.method = method
        self._explicit_labeler = labeler
        self.eraser = _ERASERS[method](**eraser_kwargs)

    def _get_labeler(self, **labeler_kwargs):
        if self._explicit_labeler is not None:
            return self._explicit_labeler
        if self.concept is None:
            raise ValueError("no labeler/concept available; pass labels=... to fit()")
        return JevLabeler(self.concept, **labeler_kwargs)

    def fit(self, embeddings, texts: Optional[Sequence[str]] = None,
            labels=None, **labeler_kwargs) -> "ConceptScrubber":
        X = np.asarray(embeddings, dtype=float)
        if labels is None:
            if texts is None:
                raise ValueError("pass either labels=... or texts=... to fit()")
            labels = self._get_labeler(**labeler_kwargs).score(texts)
        self.labels_ = np.asarray(labels, dtype=float)
        if len(self.labels_) != len(X):
            raise ValueError("labels and embeddings must have the same length")
        self.eraser.fit(X, self.labels_)
        return self

    def transform(self, embeddings) -> np.ndarray:
        return self.eraser.transform(np.asarray(embeddings, dtype=float))

    def fit_transform(self, embeddings, texts=None, labels=None, **labeler_kwargs) -> np.ndarray:
        self.fit(embeddings, texts=texts, labels=labels, **labeler_kwargs)
        return self.transform(embeddings)

    def audit(self, embeddings, texts: Optional[Sequence[str]] = None,
              labels=None, **labeler_kwargs) -> dict:
        """Concept recoverability (probe AUC) before vs. after erasure on this data."""
        X = np.asarray(embeddings, dtype=float)
        if labels is None:
            if texts is not None:
                labels = self._get_labeler(**labeler_kwargs).score(texts)
            elif hasattr(self, "labels_") and len(self.labels_) == len(X):
                labels = self.labels_
            else:
                raise ValueError("pass labels=... or texts=... to audit()")
        y = np.asarray(labels, dtype=float)
        return {
            "concept_auc_before": concept_auc(X, y),
            "concept_auc_after": concept_auc(self.transform(X), y),
        }
