"""Per-document entity extraction for high-cardinality concept erasure.

For a **many-valued identity** (occupation, topic, nationality, product category) a single scorer
under-spans the concept, so scoring a handful of questions only erases the common/head values. The
robust alternative is to **read the attribute value off each document** with an LLM
(open-vocabulary), one-hot the extracted values, and hand that label matrix to LEACE.

This discovers the full vocabulary actually present -- *including the rare tail* -- and erases far
more of the concept than scored questions (empirically ~0.64 vs ~0.80 one-vs-rest AUC on Bias in
Bios occupation). It is used by :class:`~jevu.scrubber.ConceptScrubber` when ``binary=False``.
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

logger = logging.getLogger(__name__)

__all__ = ["EntityExtractor"]

EXTRACT_SYSTEM = (
    "You read a text and extract the value of one attribute. Reply with a JSON object "
    '{{"value": "..."}} where value is the attribute\'s value as a short, canonical, lowercase '
    'noun phrase (1-3 words), or "none" if the text does not express it. Merge obvious variants to a '
    "canonical form (e.g. doctor/physician/MD -> physician). Attribute: {concept}."
)


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class EntityExtractor:
    """Extract a per-document attribute value with an LLM, then one-hot it for LEACE.

    ``score(texts)`` returns an ``(n_texts, n_distinct_values)`` one-hot matrix: the label matrix that
    :class:`~jevu.erasers.LeaceEraser` erases. Extraction happens only at ``fit`` time; the fitted
    eraser is a fixed affine map applied to new embeddings with no further LLM calls.

    Parameters
    ----------
    concept : the attribute to read off each text, e.g. ``"occupation"`` / ``"the person's profession"``.
    llm_model : chat model used for extraction.
    openai_api_key : key (else ``$OPENAI_API_KEY``).
    cache_dir : if set, each ``(model, concept, text)`` extraction is cached on disk.
    max_workers : concurrency for the per-document calls.
    temperature : 0 for deterministic, reproducible extraction.
    llm_client : optional injected client (for tests).
    """

    def __init__(self, concept: str, *, llm_model: str = "gpt-4o-mini",
                 openai_api_key: Optional[str] = None, cache_dir: Optional[str] = None,
                 max_workers: int = 16, temperature: float = 0.0, llm_client=None,
                 progress: bool = True):
        if not concept or not concept.strip():
            raise ValueError("concept must be a non-empty string")
        self.concept = concept.strip()
        self.llm_model = llm_model
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.temperature = temperature
        self.llm_client = llm_client
        self.progress = progress
        self._client_obj = None

    # -- internals -------------------------------------------------------------
    def _client(self):
        if self.llm_client is not None:
            return self.llm_client
        if self._client_obj is None:
            if not self.openai_api_key:
                raise ValueError("no cached extraction and no OpenAI key/client to extract values")
            from openai import OpenAI
            self._client_obj = OpenAI(api_key=self.openai_api_key, timeout=60)
        return self._client_obj

    def _cache_path(self, text: str):
        if not self.cache_dir:
            return None
        return self.cache_dir / f"entity_{_digest([self.llm_model, self.concept, text])}.json"

    def _extract_one(self, text: str) -> str:
        path = self._cache_path(text)
        if path and path.exists():
            return json.loads(path.read_text())
        system = EXTRACT_SYSTEM.format(concept=self.concept)
        resp = self._client().chat.completions.create(
            model=self.llm_model, temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": text[:800]}])
        val = json.loads(resp.choices[0].message.content).get("value", "none")
        val = (val or "none").strip().lower() or "none"
        if path:
            path.write_text(json.dumps(val))
        return val

    # -- public API ------------------------------------------------------------
    def extract(self, texts: Sequence[str]) -> list:
        """Return the extracted attribute value (a string) for each text."""
        texts = list(texts)
        n_missing = sum(1 for t in texts
                        if not (self.cache_dir and self._cache_path(t).exists()))
        logger.info("extracting %r from %d texts (%d cached, %d to call)",
                    self.concept, len(texts), len(texts) - n_missing, n_missing)
        out: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {pool.submit(self._extract_one, t): i for i, t in enumerate(texts)}
            for fut in progress(as_completed(futs), total=len(texts),
                                desc=f"extract: {self.concept[:30]}",
                                enabled=self.progress and len(texts) > 1):
                out[futs[fut]] = fut.result()
        self.values_ = [out[i] for i in range(len(texts))]
        return self.values_

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Extract per-document values and return their ``(n_texts, n_distinct_values)`` one-hot."""
        vals = self.extract(texts)
        uniq = sorted(set(vals))
        self.vocabulary_ = uniq
        idx = {u: i for i, u in enumerate(uniq)}
        M = np.zeros((len(vals), len(uniq)), dtype=float)
        for i, v in enumerate(vals):
            M[i, idx[v]] = 1.0
        logger.info("extracted %d distinct values -> one-hot %s", len(uniq), M.shape)
        return M
