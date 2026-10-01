import numpy as np

from jevu import OpenAIEmbedder


class _Item:
    def __init__(self, e): self.embedding = e


class _Resp:
    def __init__(self, data): self.data = data


class FakeClient:
    """Minimal stand-in for the OpenAI client: client.embeddings.create(...)."""
    def __init__(self):
        self.embeddings = self
        self.calls = 0

    def create(self, model, input):
        self.calls += 1
        return _Resp([_Item([float(len(t)), 1.0, 2.0]) for t in input])


def test_embed_shape_and_values(tmp_path):
    client = FakeClient()
    emb = OpenAIEmbedder(api_key="k", client=client, cache_dir=str(tmp_path))
    v = emb.embed(["a", "bb", "ccc"])
    assert v.shape == (3, 3)
    assert list(v[:, 0]) == [1.0, 2.0, 3.0]   # len-based fake embedding


def test_embeddings_are_cached(tmp_path):
    client = FakeClient()
    emb = OpenAIEmbedder(api_key="k", client=client, cache_dir=str(tmp_path))
    emb.embed(["a", "b"])
    first = client.calls
    emb.embed(["a", "b"])                     # all cached -> no new API calls
    assert client.calls == first
