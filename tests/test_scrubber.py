import numpy as np
import pytest

from jevu import ConceptScrubber, concept_auc


def make_data(n=800, d=24, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=n)
    X = rng.standard_normal((n, d))
    d0 = np.zeros(d); d0[0] = 1.0
    X += 4.0 * np.outer(z, d0)
    return X, z


class FakeLabeler:
    """Stand-in for JevLabeler: scores by keyword, no network."""
    def score(self, texts):
        return np.array([1.0 if "woman" in t.lower() else 0.0 for t in texts])


class FakeEmbedder:
    """Deterministic embedder that plants the concept on dim 0, no network."""
    def embed(self, texts):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((len(texts), 16))
        for i, t in enumerate(texts):
            if "woman" in t.lower():
                X[i, 0] += 5.0
        return X


def test_fit_with_explicit_labels_and_embeddings():
    X, z = make_data()
    scr = ConceptScrubber(method="leace")          # no concept/labeler needed when labels given
    Xc = scr.fit_transform(embeddings=X, labels=z)
    assert concept_auc(Xc, z) < 0.65


def test_text_first_embeds_and_labels_internally():
    texts = (["a woman leads the team"] * 50) + (["a man leads the team"] * 50)
    scr = ConceptScrubber(concept="woman?", method="inlp",
                          labeler=FakeLabeler(), embedder=FakeEmbedder())
    scr.fit(texts)                                  # embeds + labels internally
    report = scr.audit(texts)
    assert report["concept_auc_before"] > 0.9
    assert report["concept_auc_after"] < 0.7


def test_byo_embeddings_with_concept_labeler():
    X, z = make_data(n=120, d=16)
    texts = ["a woman" if zi else "a man" for zi in z]
    scr = ConceptScrubber(method="leace", labeler=FakeLabeler())
    scr.fit(texts=texts, embeddings=X)             # your embeddings, labeler scores the concept
    assert concept_auc(scr.transform(embeddings=X), z) < 0.65


def test_concept_scores_delegates_to_labeler():
    scr = ConceptScrubber(labeler=FakeLabeler())
    s = scr.concept_scores(["a woman engineer", "a man"])
    assert s[0] > 0.5 > s[1]


def test_needs_labeler_or_labels():
    scr = ConceptScrubber()                         # no concept, no labeler
    with pytest.raises(ValueError):
        scr.fit(texts=["anything"])                 # nothing can produce labels


def test_bad_method():
    with pytest.raises(ValueError):
        ConceptScrubber(concept="x", method="nope")
