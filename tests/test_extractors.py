import json
import types

import numpy as np

from jevu import ConceptScrubber, EntityExtractor


def fake_llm(fn):
    """Stand-in OpenAI client: value = fn(user_text)."""
    def create(**kw):
        text = kw["messages"][-1]["content"]
        content = json.dumps({"value": fn(text)})
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=content))])
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def _occ(t):
    return "nurse" if "nurse" in t else ("engineer" if "engineer" in t else "poet")


def test_extract_onehot(tmp_path):
    ex = EntityExtractor("occupation", llm_client=fake_llm(_occ), cache_dir=str(tmp_path))
    M = ex.score(["a nurse", "an engineer", "a nurse too"])
    assert M.shape == (3, 2)                       # nurse, engineer -> 2 distinct values
    assert ex.vocabulary_ == ["engineer", "nurse"]  # sorted
    assert M[0, ex.vocabulary_.index("nurse")] == 1.0
    assert M[1, ex.vocabulary_.index("engineer")] == 1.0


def test_extract_caches(tmp_path):
    calls = {"n": 0}
    def fn(t):
        calls["n"] += 1
        return "x"
    ex = EntityExtractor("c", llm_client=fake_llm(fn), cache_dir=str(tmp_path))
    ex.extract(["a", "b"])
    n1 = calls["n"]
    ex.extract(["a", "b"])                 # second run hits the disk cache
    assert calls["n"] == n1 == 2


def test_empty_concept_raises():
    import pytest
    with pytest.raises(ValueError):
        EntityExtractor("  ")


def test_scrubber_binary_false_erases_identity():
    # end-to-end: binary=False -> extraction one-hot -> LEACE removes the 3-way identity
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    texts = (["a nurse"] * 40) + (["an engineer"] * 40) + (["a poet"] * 40)
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)
    X = rng.standard_normal((120, 16))
    for c in range(3):
        X[y == c, c] += 5.0                 # each class separable on its own axis
    # inject the extractor (so no network) via labeler=
    ex = EntityExtractor("occupation", llm_client=fake_llm(_occ))
    scr = ConceptScrubber(method="leace", labeler=ex)
    Xc = scr.fit_transform(X, texts=texts)
    acc_before = LogisticRegression(max_iter=1000).fit(X, y).score(X, y)
    acc_after = LogisticRegression(max_iter=1000).fit(Xc, y).score(Xc, y)
    assert acc_before > 0.95
    assert acc_after < 0.6                   # identity largely erased
    assert scr.labels_.shape == (120, 3)     # eraser fit on the 3-value one-hot


def test_scrubber_binary_false_builds_extractor():
    # binary=False routes _labeler_ to an EntityExtractor (no labeler injected)
    scr = ConceptScrubber(concept="occupation", binary=False, openai_api_key="k")
    lab = scr._labeler_()
    assert isinstance(lab, EntityExtractor)
