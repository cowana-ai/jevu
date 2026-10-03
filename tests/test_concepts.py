import json
import types

import httpx
import numpy as np

from jevu import ConceptScrubber, LLMConceptLabeler, concept_auc


def fake_llm(questions):
    """A stand-in OpenAI chat client whose completion returns the given questions as JSON."""
    content = json.dumps({"questions": questions})
    create = lambda **kw: types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def jev_mock(score_for):
    def handler(request: httpx.Request) -> httpx.Response:
        doc = json.loads(request.content)["state"]["document"]
        q = json.loads(request.content)["questions"]["q1"]["instructions"]
        return httpx.Response(200, json={"answers": {"q1": {"noul": score_for(doc, q)}}})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_expands_concept_and_scores_matrix(tmp_path):
    qs = ["Does the text refer to a woman?", "Does the text refer to a man?"]
    # JEV returns high score when the question's subject word appears in the text
    def score_for(doc, q):
        if "woman" in q and "woman" in doc: return 0.9
        if "man" in q and "woman" not in doc and "man" in doc: return 0.9
        return 0.1
    lab = LLMConceptLabeler("gender", n_questions=2, questions=None,
                            llm_client=fake_llm(qs), jev_client=jev_mock(score_for),
                            openai_api_key="k", openrouter_api_key="k", cache_dir=str(tmp_path))
    assert lab.questions() == qs
    M = lab.score(["a woman led", "a man led"])
    assert M.shape == (2, 2)
    assert M[0, 0] > 0.8 and M[1, 1] > 0.8     # woman-col fires on bio 0, man-col on bio 1


def test_expansion_prompt_handles_terse_and_discriminative():
    from jevu.concepts import EXPANSION_SYSTEM
    s = EXPANSION_SYSTEM.lower()
    assert ("terse" in s) or ("ambiguous" in s)       # robust to short/ambiguous input
    assert "cover" in s                                # many-valued identity -> covering sub-categories
    assert ("binary" in s) and ("identity" in s)      # branches on cardinality
    assert "yes/no" in s


def test_terse_concept_expands_via_llm(tmp_path):
    import json, types, httpx
    qs = ["Does the text refer to a woman or female person?",
          "Does the text refer to a man or male person?"]
    llm = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
        create=lambda **kw: types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=json.dumps({"questions": qs}))) ]))))
    jev = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"answers": {"q1": {"noul": 0.5}}})))
    lab = LLMConceptLabeler("genders", n_questions=2, llm_client=llm, jev_client=jev,
                            openai_api_key="k", openrouter_api_key="k", cache_dir=str(tmp_path))
    assert lab.questions() == qs                      # terse "genders" -> concrete questions
    assert lab.score(["a woman led"]).shape == (1, 2)


def test_auto_expand_lets_llm_choose_count(tmp_path):
    import types, httpx
    # LLM returns a variable-length list + a cardinality field (auto mode)
    body = json.dumps({"cardinality": "identity",
                       "questions": ["Does the text refer to a nurse?", "Does the text refer to a teacher?",
                                     "Does the text refer to an engineer?"]})
    llm = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
        create=lambda **kw: types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=body))]))))
    jev = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"answers": {"q1": {"noul": 0.3}}})))
    lab = LLMConceptLabeler("occupation", n_questions="auto", max_questions=24,
                            llm_client=llm, jev_client=jev, openai_api_key="k",
                            openrouter_api_key="k", cache_dir=str(tmp_path))
    qs = lab.questions()
    assert len(qs) == 3                      # LLM chose the count, not a fixed n
    assert lab.score(["a nurse"]).shape == (1, 3)


def test_bad_n_questions():
    import pytest
    with pytest.raises(ValueError):
        LLMConceptLabeler("x", n_questions=0)


def test_explicit_questions_skip_llm():
    lab = LLMConceptLabeler("gender", questions=["Does the text refer to a woman?"])
    assert lab.questions() == ["Does the text refer to a woman?"]   # no client needed


def test_scrubber_expand_multi_concept_erasure():
    # end-to-end with an injected multi-column labeler (no network)
    class MultiLabeler:
        def score(self, texts):
            w = np.array([1.0 if "woman" in t else 0.0 for t in texts])
            m = np.array([1.0 if "man" in t and "woman" not in t else 0.0 for t in texts])
            return np.column_stack([w, m])

    rng = np.random.default_rng(0)
    texts = (["a woman led"] * 60) + (["a man led"] * 60)
    X = rng.standard_normal((120, 20))
    X[:, 0] += 5.0 * np.array([1.0 if "woman" in t else 0.0 for t in texts])
    X[:, 1] += 5.0 * np.array([1.0 if ("man" in t and "woman" not in t) else 0.0 for t in texts])
    scr = ConceptScrubber(method="leace", labeler=MultiLabeler())
    scr.fit(X, texts=texts)
    rep = scr.audit(X, texts=texts)
    assert rep["n_concepts"] == 2
    assert rep["concept_auc_before"] > 0.9 and rep["concept_auc_after"] < 0.7
