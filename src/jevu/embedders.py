"""Turn texts into embeddings. A thin, cached OpenAI embedder is the default so the
high-level API can accept raw text; bring any object with an ``.embed(texts) -> np.ndarray``
method to use a different model.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

__all__ = ["OpenAIEmbedder"]


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class OpenAIEmbedder:
    """Embed texts with an OpenAI embedding model, caching each text on disk.

    Parameters
    ----------
    model : embedding model id.
    api_key : OpenAI key; falls back to ``$OPENAI_API_KEY``.
    cache_dir : if set, each text's vector is cached so re-runs cost nothing.
    batch_size : how many texts per API request.
    client : optional pre-built OpenAI client (handy for tests / custom endpoints).
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None, batch_size: int = 200, timeout: float = 90.0,
                 client=None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.timeout = timeout
        self._client = client

    def _path(self, text: str):
        if not self.cache_dir:
            return None
        return self.cache_dir / f"emb_{_digest({'m': self.model, 't': text})}.json"

    def _make_client(self):
        from openai import OpenAI

        return OpenAI(api_key=self.api_key, timeout=self.timeout)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        out: dict[int, np.ndarray] = {}
        missing = []
        for i, t in enumerate(texts):
            p = self._path(t)
            if p and p.exists():
                out[i] = np.asarray(json.loads(p.read_text()), dtype=float)
            else:
                missing.append(i)
        if missing:
            if not self.api_key and self._client is None:
                raise ValueError("no cached embedding and no api_key/OPENAI_API_KEY set")
            client = self._client or self._make_client()
            for s in range(0, len(missing), self.batch_size):
                idx = missing[s:s + self.batch_size]
                resp = client.embeddings.create(model=self.model, input=[texts[i] for i in idx])
                for i, item in zip(idx, resp.data):
                    v = np.asarray(item.embedding, dtype=float)
                    out[i] = v
                    p = self._path(texts[i])
                    if p:
                        p.write_text(json.dumps(v.tolist()))
        return np.vstack([out[i] for i in range(len(texts))])
