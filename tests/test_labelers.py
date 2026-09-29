import httpx
import numpy as np

from jevu import JevLabeler


def _mock_client(score_for):
    def handler(request: httpx.Request) -> httpx.Response:
        import json
        doc = json.loads(request.content)["state"]["document"]
        return httpx.Response(200, json={"answers": {"q1": {"noul": score_for(doc)}}})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_score_and_label_with_mock_client(tmp_path):
    client = _mock_client(lambda doc: 0.9 if "woman" in doc.lower() else 0.1)
    lab = JevLabeler("Does the text describe a woman?", api_key="k",
                     client=client, cache_dir=str(tmp_path))
    texts = ["She is a woman engineer", "He fixed the car"]
    scores = lab.score(texts)
    assert scores.shape == (2,)
    assert scores[0] > 0.8 and scores[1] < 0.2
    assert list(lab.label(texts)) == [1, 0]


def test_scores_are_cached(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"answers": {"q1": {"noul": 0.7}}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    lab = JevLabeler("concept?", api_key="k", client=client, cache_dir=str(tmp_path))
    lab.score(["a", "b"])
    n_after_first = calls["n"]
    lab.score(["a", "b"])                 # second run should hit the disk cache
    assert calls["n"] == n_after_first == 2


def test_missing_key_without_cache_raises():
    import pytest
    lab = JevLabeler("concept?", api_key=None)
    with pytest.raises(ValueError):
        lab.score(["anything"])
