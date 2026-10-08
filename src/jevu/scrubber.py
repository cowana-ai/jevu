"""High-level API: give a concept and your embeddings, erase the concept.

Bring embeddings from **any** model — this package never embeds for you. Give it the embeddings plus
either the raw texts (so laya can label the concept) or your own labels::

    from jevu import ConceptScrubber

    scrubber = ConceptScrubber(concept="Does the text describe a woman?")
    scrubber.fit(X, texts=texts)         # your embeddings; laya labels the concept locally
    X_clean = scrubber.transform(X_new)  # erase the concept from new embeddings

    scrubber.fit(X, labels=y)            # your embeddings and labels (no scorer at all)
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from .audit import concept_auc
from .concepts import LLMConceptLabeler
from .erasers import InlpEraser, LeaceEraser
from .extractors import EntityExtractor
from .laya_labeler import LayaLabeler

logger = logging.getLogger(__name__)

__all__ = ["ConceptScrubber"]

_ERASERS = {"leace": LeaceEraser, "inlp": InlpEraser}


def _greedy_select(Z: np.ndarray, k: int, questions: Sequence[str]) -> list:
    """Greedy forward selection: pick the k questions whose columns best reconstruct the whole
    pool of scores Z (label-free). Each step adds the column that most reduces the residual of
    least-squares reconstructing Z from the selected columns. Logs every pick."""
    Zc = Z - Z.mean(axis=0)
    P = Zc.shape[1]
    selected: list = []
    for step in range(min(k, P)):
        best = None
        for c in range(P):
            if c in selected:
                continue
            A = Zc[:, selected + [c]]
            coef = np.linalg.pinv(A) @ Zc           # A⁺ Z: least-squares coeffs via SVD-based pseudoinverse
            resid = float(np.linalg.norm(Zc - A @ coef))
            if best is None or resid < best[0]:
                best = (resid, c)
        selected.append(best[1])
        logger.info("greedy select %d/%d: %r (reconstruction residual %.3f)",
                    step + 1, k, questions[best[1]], best[0])
    return selected


class ConceptScrubber:
    """Erase a target concept from embeddings you provide.

    Two paths, chosen by ``binary`` (the right one depends on the concept's *rank*):

    * ``binary=True`` (default) -- **low-rank / binary concepts** (gender, sentiment, tone). The LLM
      writes 2-3 yes/no questions (or you pass one via ``expand=False``), **laya** scores them
      locally, and LEACE erases. One-two questions span the concept, so this drives the probe to
      chance -- laya's sweet spot (free, local, offline).
    * ``binary=False`` -- **high-cardinality identities** (occupation, topic, nationality). The LLM
      **extracts** the attribute value from each document (open-vocabulary), the values are one-hot
      encoded, and LEACE erases. This discovers the full vocabulary present -- including the rare
      tail -- and erases far more of the concept than scored questions.

    Parameters
    ----------
    concept : the concept to erase. A yes/no question or attribute word for ``binary=True``; the
        attribute to read off each document (e.g. ``"occupation"``) for ``binary=False``.
    binary : pick the path above. ``True`` = laya-scored questions; ``False`` = LLM entity extraction.
    method : ``"leace"`` (default, closed-form, minimal-damage) or ``"inlp"`` (iterative).
    expand : only for ``binary=True``. ``False`` (score the concept as one question), ``True`` (LLM
        expands into ``n_questions`` sub-questions), or ``"auto"`` (the LLM decides how many).
    n_questions : number of sub-questions when ``binary=True, expand=True``; the cap when ``"auto"``.
    select_k : if set (with expansion), greedily keep only the ``select_k`` questions that best
        reconstruct the full pool, and erase just those -- a cheap, reusable eraser. The picks are
        logged and stored in ``selected_questions_``.
    select_sample : run the (expensive) selection on a random subset of this many rows -- the pool is
        scored only on the sample, then just the ``select_k`` winners are scored on the full data.
        Cuts selection cost from ``pool x n`` to ``pool x select_sample + select_k x n``.
    labeler : object with ``.score(texts) -> (n,) or (n, k)``; overrides ``concept``/``expand``.
    laya_model, llm_model : model ids for laya scoring / concept expansion.
    device : torch device for laya (``None`` = auto, e.g. ``mps``/``cuda``/``cpu``).
    cache_dir : shared on-disk cache for laya scores and concept expansions.
    openai_api_key : key for the expansion LLM (only needed when ``expand`` is truthy).
    eraser_kwargs : forwarded to the chosen eraser.
    """

    def __init__(self, concept: Optional[str] = None, method: str = "leace", *,
                 binary: bool = True, expand=True, n_questions: int = 3, max_questions: int = 24,
                 select_k: Optional[int] = None, select_sample: Optional[int] = None,
                 random_state: int = 0, labeler=None,
                 laya_model: str = "convaiinnovations/laya", llm_model: str = "gpt-4o-mini",
                 device: Optional[str] = None, cache_dir: Optional[str] = None, batch_size: int = 16,
                 max_workers: int = 16, openai_api_key: Optional[str] = None,
                 **eraser_kwargs):
        if method not in _ERASERS:
            raise ValueError(f"method must be one of {sorted(_ERASERS)}")
        self.concept = concept
        self.method = method
        self.binary = binary
        self.expand = expand
        self.n_questions = n_questions
        self.max_questions = max_questions
        self.select_k = select_k
        self.select_sample = select_sample
        self.random_state = random_state
        self._labeler = labeler
        self.laya_model = laya_model
        self.llm_model = llm_model
        self.device = device
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._openai_api_key = openai_api_key
        self.eraser = _ERASERS[method](**eraser_kwargs)

    def _labeler_(self):
        if self._labeler is None:
            if self.concept is None:
                raise ValueError("no labels given and no concept/labeler set; "
                                 "pass concept=..., labeler=..., or labels=...")
            if not self.binary:
                # high-cardinality path: LLM reads the attribute off each doc -> one-hot -> LEACE
                self._labeler = EntityExtractor(
                    self.concept, llm_model=self.llm_model, openai_api_key=self._openai_api_key,
                    cache_dir=self.cache_dir, max_workers=self.max_workers)
            elif self.expand:
                # binary/low-rank path: LLM writes 2-3 questions, laya scores them
                nq = "auto" if self.expand == "auto" else self.n_questions
                self._labeler = LLMConceptLabeler(
                    self.concept, n_questions=nq, max_questions=self.max_questions,
                    llm_model=self.llm_model, laya_model=self.laya_model,
                    openai_api_key=self._openai_api_key, device=self.device,
                    cache_dir=self.cache_dir, batch_size=self.batch_size)
            else:
                # binary path, single explicit question: laya scores the concept directly
                self._labeler = LayaLabeler(self.concept, model=self.laya_model, device=self.device,
                                            cache_dir=self.cache_dir, batch_size=self.batch_size)
        return self._labeler

    def _score_pool(self, texts):
        """Score the concept once -> ``(Z, questions)``. For an expanded concept ``Z`` is the full
        ``(n, pool)`` matrix and ``questions`` is the pool; for a single concept ``Z`` is ``(n,)`` and
        ``questions`` is ``None``. Memoized on the instance so repeated selections never re-score."""
        labeler = self._labeler_()
        pool = labeler.questions(texts) if hasattr(labeler, "questions") else None
        if pool is not None and hasattr(labeler, "score_questions"):
            Z = np.asarray(labeler.score_questions(pool, texts), dtype=float)
        else:
            Z = np.asarray(labeler.score(texts), dtype=float)
        if pool is not None:
            self.concept_questions_ = pool
        return Z, pool

    def _select(self, Z, questions, k, sample_idx=None) -> list:
        """Greedily pick ``k`` columns of an already-scored matrix ``Z`` (no scoring). ``sample_idx``
        restricts the greedy criterion to a subset of rows. Returns the selected column indices."""
        M = Z if sample_idx is None else Z[sample_idx]
        return _greedy_select(M, k, questions)

    def _labels_for(self, X, texts, labels) -> np.ndarray:
        if labels is None:
            if texts is None:
                raise ValueError("pass labels=... or texts=... so the concept can be scored")
            labeler = self._labeler_()
            pool = labeler.questions(texts) if hasattr(labeler, "questions") else None
            can_select = (self.select_k and pool is not None and self.select_k < len(pool))

            if can_select and self.select_sample and hasattr(labeler, "score_questions") \
                    and self.select_sample < len(texts):
                # cheapest path: score the pool only on a SAMPLE, pick k, score k on the full data.
                self.concept_questions_ = pool
                rng = np.random.default_rng(self.random_state)
                sub = rng.choice(len(texts), self.select_sample, replace=False)
                logger.info("selecting %d/%d questions on a %d-row sample", self.select_k, len(pool), len(sub))
                Zsub = np.asarray(labeler.score_questions(pool, [texts[i] for i in sub]), dtype=float)
                idx = self._select(Zsub, pool, self.select_k)
                self.selected_questions_ = [pool[i] for i in idx]
                logger.info("kept %d/%d questions: %s", len(idx), len(pool), self.selected_questions_)
                labels = labeler.score_questions(self.selected_questions_, texts)
            else:
                labels, pool = self._score_pool(texts)      # one scoring pass
                if can_select:
                    idx = self._select(labels, pool, self.select_k)
                    self.selected_questions_ = [pool[i] for i in idx]
                    logger.info("kept %d/%d questions: %s", len(idx), labels.shape[1], self.selected_questions_)
                    labels = labels[:, idx]
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
