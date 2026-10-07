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
    """Stand-in labeler: scores by keyword, no model."""
    def score(self, texts):
        return np.array([1.0 if "woman" in t.lower() else 0.0 for t in texts])


def test_fit_with_explicit_labels():
    X, z = make_data()
    scr = ConceptScrubber(method="leace")          # no concept/labeler needed when labels given
    Xc = scr.fit_transform(X, labels=z)
    assert concept_auc(Xc, z) < 0.65


def test_labels_via_labeler_byo_embeddings():
    texts = (["a woman leads the team"] * 50) + (["a man leads the team"] * 50)
    z = np.array([1] * 50 + [0] * 50)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((100, 16)); X[:, 0] += 5.0 * z   # bring your own embeddings
    scr = ConceptScrubber(method="inlp", labeler=FakeLabeler())
    scr.fit(X, texts=texts)                         # labeler scores the texts; X is provided
    report = scr.audit(X, texts=texts)
    assert report["concept_auc_before"] > 0.9
    assert report["concept_auc_after"] < 0.7


def test_concept_scores_delegates_to_labeler():
    scr = ConceptScrubber(labeler=FakeLabeler())
    s = scr.concept_scores(["a woman engineer", "a man"])
    assert s[0] > 0.5 > s[1]


def test_needs_labeler_or_labels():
    X, _ = make_data(n=20, d=8)
    scr = ConceptScrubber()                         # no concept, no labeler
    with pytest.raises(ValueError):
        scr.fit(X)                                  # nothing can produce labels


def test_bad_method():
    with pytest.raises(ValueError):
        ConceptScrubber(concept="x", method="nope")
