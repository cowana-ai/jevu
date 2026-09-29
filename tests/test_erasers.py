import numpy as np
import pytest

from jevu import InlpEraser, LeaceEraser, concept_auc


def make_data(n=800, d=24, seed=0):
    """Embeddings where a binary concept z lives mostly on one direction and an
    independent task signal t lives on another, plus noise."""
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=n)
    t = rng.integers(0, 2, size=n)
    X = rng.standard_normal((n, d))
    concept_dir = np.zeros(d); concept_dir[0] = 1.0
    task_dir = np.zeros(d); task_dir[1] = 1.0
    X += 4.0 * np.outer(z, concept_dir)   # concept signal
    X += 4.0 * np.outer(t, task_dir)      # unrelated task signal
    return X, z, t


@pytest.mark.parametrize("Eraser", [InlpEraser, LeaceEraser])
def test_erasure_removes_concept_keeps_task(Eraser):
    X, z, t = make_data()
    Xc = Eraser().fit_transform(X, z)
    # concept becomes ~unrecoverable
    assert concept_auc(X, z) > 0.9
    assert concept_auc(Xc, z) < 0.65
    # an independent task signal is preserved
    assert concept_auc(Xc, t) > 0.85


def test_leace_zeroes_cross_covariance():
    X, z, _ = make_data()
    Xc = LeaceEraser().fit_transform(X, z)
    zc = z - z.mean()
    Xcc = Xc - Xc.mean(axis=0)
    cross = (Xcc.T @ zc) / len(z)
    assert np.abs(cross).max() < 1e-6   # LEACE guarantee: exact linear erasure


def test_transform_applies_to_unseen_rows():
    X, z, _ = make_data()
    er = LeaceEraser().fit(X[:400], z[:400])
    out = er.transform(X[400:])
    assert out.shape == X[400:].shape


def test_inlp_stops_when_no_concept():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 10))
    z = np.zeros(200, dtype=int)   # single class -> nothing to erase
    er = InlpEraser().fit(X, z)
    assert er.n_iters_ == 0
    assert np.allclose(er.transform(X), X)
