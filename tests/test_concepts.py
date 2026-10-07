import json
import types

import numpy as np

from jevu import ConceptScrubber, LLMConceptLabeler, concept_auc


def fake_llm(questions):
    """A stand-in OpenAI chat client whose completion returns the given questions as JSON."""
    content = json.dumps({"questions": questions})
    create = lambda **kw: types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def fake_laya(score_for):
    """A stand-in laya agent: answers every noul question for each state in one batch."""
    def predict_batch(states, questions, batch_size=None, **kw):
        return [{"answers": {qid: {"noul": score_for(s, q["instructions"])}
                             for qid, q in questions.items()}} for s in states]
    return types.SimpleNamespace(predict_batch=predict_batch)


def test_expands_concept_and_scores_matrix(tmp_path):
    qs = ["Does the text refer to a woman?", "Does the text refer to a man?"]
    # laya returns high score when the question's subject word appears in the text
    def score_for(doc, q):
        if "woman" in q and "woman" in doc: return 0.9
        if "man" in q and "woman" not in doc and "man" in doc: return 0.9
        return 0.1
    lab = LLMConceptLabeler("gender", n_questions=2, questions=None,
                            llm_client=fake_llm(qs), laya_agent=fake_laya(score_for),
                            openai_api_key="k", cache_dir=str(tmp_path))
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
    qs = ["Does the text refer to a woman or female person?",
          "Does the text refer to a man or male person?"]
    lab = LLMConceptLabeler("genders", n_questions=2, llm_client=fake_llm(qs),
                            laya_agent=fake_laya(lambda d, q: 0.5),
                            openai_api_key="k", cache_dir=str(tmp_path))
    assert lab.questions() == qs                      # terse "genders" -> concrete questions
    assert lab.score(["a woman led"]).shape == (1, 2)


def test_auto_expand_lets_llm_choose_count(tmp_path):
    # LLM returns a variable-length list + a cardinality field (auto mode)
    body = json.dumps({"cardinality": "identity",
                       "questions": ["Does the text refer to a nurse?", "Does the text refer to a teacher?",
                                     "Does the text refer to an engineer?"]})
    llm = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
        create=lambda **kw: types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=body))]))))
    lab = LLMConceptLabeler("occupation", n_questions="auto", max_questions=24,
                            llm_client=llm, laya_agent=fake_laya(lambda d, q: 0.3),
                            openai_api_key="k", cache_dir=str(tmp_path))
    qs = lab.questions()
    assert len(qs) == 3                      # LLM chose the count, not a fixed n
    assert lab.score(["a nurse"]).shape == (1, 3)


def test_select_k_greedy_keeps_k_questions():
    # a pool of 4 questions where 2 are redundant copies; select_k=2 should keep 2
    rng = np.random.default_rng(0)
    n, d = 300, 16
    a = rng.integers(0, 2, n); b = rng.integers(0, 2, n)
    X = rng.standard_normal((n, d)); X[:, 0] += 4.0 * a; X[:, 1] += 4.0 * b

    class PoolLabeler:
        _qs = ["q_a", "q_a_copy", "q_b", "q_noise"]
        def questions(self, texts=None): return self._qs
        def score(self, texts):
            noise = rng.standard_normal(n)
            return np.column_stack([a, a, b, (noise > 0).astype(int)]).astype(float)

    scr = ConceptScrubber(method="leace", expand=True, labeler=PoolLabeler(), select_k=2)
    scr.fit(X, texts=["t"] * n)
    assert len(scr.selected_questions_) == 2
    assert scr.labels_.shape[1] == 2            # eraser fit on the 2 kept questions
    assert scr.concept_questions_ == PoolLabeler._qs


def test_select_k_with_sampling():
    # pool scored on a subset for selection; winners scored on full data
    rng = np.random.default_rng(0)
    n, d = 300, 16
    a = rng.integers(0, 2, n); b = rng.integers(0, 2, n)
    X = rng.standard_normal((n, d)); X[:, 0] += 4.0 * a; X[:, 1] += 4.0 * b

    class PoolLabeler:
        _qs = ["q_a", "q_a_copy", "q_b", "q_noise"]
        calls = {"full_rows": 0}
        def questions(self, texts=None): return self._qs
        def score_questions(self, questions, texts):
            self.calls["full_rows"] = max(self.calls["full_rows"], len(texts))
            cols = {"q_a": a, "q_a_copy": a, "q_b": b,
                    "q_noise": (rng.standard_normal(n) > 0).astype(int)}
            return np.column_stack([cols[q][:len(texts)] for q in questions]).astype(float)
        def score(self, texts): return self.score_questions(self._qs, texts)

    lab = PoolLabeler()
    scr = ConceptScrubber(method="leace", expand=True, labeler=lab, select_k=2, select_sample=120)
    scr.fit(X, texts=["t"] * n)
    assert len(scr.selected_questions_) == 2
    assert scr.labels_.shape == (n, 2)          # eraser fit on full data, 2 questions


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
