import types

import numpy as np

from jevu import LayaLabeler


def fake_agent(score_fn, counter=None):
    """Stand-in laya agent: returns noul = score_fn(state, question) for each typed question."""
    def predict_batch(states, questions, batch_size=None, **kw):
        out = []
        for s in states:
            ans = {}
            for qid, q in questions.items():
                if counter is not None:
                    counter["n"] += 1
                ans[qid] = {"noul": score_fn(s, q["instructions"])}
            out.append({"answers": ans})
        return out
    return types.SimpleNamespace(predict_batch=predict_batch)


def test_score_single_question(tmp_path):
    ag = fake_agent(lambda doc, q: 0.9 if "woman" in doc.lower() else 0.1)
    lab = LayaLabeler("Does the text describe a woman?", agent=ag, cache_dir=str(tmp_path))
    s = lab.score(["She is a woman engineer", "He fixed the car"])
    assert s.shape == (2,)
    assert s[0] > 0.8 and s[1] < 0.2
    assert list(lab.label(["She is a woman engineer", "He fixed the car"])) == [1, 0]


def test_score_multi_question_matrix(tmp_path):
    def sf(doc, q):
        return 0.9 if (("woman" in q and "woman" in doc) or ("man" in q and "woman" not in doc and "man" in doc)) else 0.1
    lab = LayaLabeler(questions=["Does the text refer to a woman?", "Does the text refer to a man?"],
                      agent=fake_agent(sf), cache_dir=str(tmp_path))
    M = lab.score(["a woman led", "a man led"])
    assert M.shape == (2, 2)
    assert M[0, 0] > 0.8 and M[1, 1] > 0.8


def test_scores_are_cached(tmp_path):
    calls = {"n": 0}
    ag = fake_agent(lambda doc, q: 0.7, counter=calls)
    lab = LayaLabeler("concept?", agent=ag, cache_dir=str(tmp_path))
    lab.score(["a", "b"])
    n_after_first = calls["n"]
    lab.score(["a", "b"])                 # second run should hit the disk cache
    assert calls["n"] == n_after_first == 2


def test_cached_needs_no_agent(tmp_path):
    # populate the cache with an agent
    LayaLabeler("c?", agent=fake_agent(lambda d, q: 0.3), cache_dir=str(tmp_path)).score(["a", "b"])
    # now fully cached: no agent -> still works (never loads the model)
    lab = LayaLabeler("c?", cache_dir=str(tmp_path), progress=False)
    assert np.allclose(lab.score(["a", "b"]), 0.3)


def test_empty_concept_raises():
    import pytest
    with pytest.raises(ValueError):
        LayaLabeler("   ")
