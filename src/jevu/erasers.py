"""Linear concept erasers that operate on embedding matrices.

Two methods are provided, both with a scikit-learn-style ``fit`` / ``transform`` API:

* :class:`InlpEraser` -- Iterative Nullspace Projection (Ravfogel et al., ACL 2020,
  "Null It Out"). Repeatedly fits a linear probe for the concept and projects the
  representations onto the null space of that probe until the concept is no longer
  linearly recoverable.
* :class:`LeaceEraser` -- LEAst-squares Concept Erasure (Belrose et al., NeurIPS 2023).
  A closed-form affine map that makes the cross-covariance between the erased
  representation and the concept exactly zero, with the smallest possible edit.

Both learn the transform on ``fit`` data and apply the *same* transform to unseen
embeddings on ``transform`` -- so you fit once and scrub a stream in production.
"""
from __future__ import annotations

import numpy as np

__all__ = ["InlpEraser", "LeaceEraser"]


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected a 2D (n_samples, n_features) array, got shape {X.shape}")
    return X


def _binarize(y) -> np.ndarray:
    """Turn concept scores into 0/1 labels. Continuous scores are thresholded at 0.5
    (JEV `noul` scores are calibrated probabilities, so 0.5 is the natural boundary)."""
    y = np.asarray(y)
    if y.dtype.kind == "f" and (y.min() < 0 or y.max() > 1 or np.unique(y).size > 2):
        return (y >= 0.5).astype(int)
    return y.astype(int)


class InlpEraser:
    """Iterative Nullspace Projection.

    Parameters
    ----------
    max_iters : maximum number of projection rounds.
    tol_auc : stop early once a probe can no longer separate the concept beyond this
        train ROC-AUC (0.5 = chance).
    C : inverse regularization strength for the logistic-regression probes.
    random_state : seed for the probes.
    """

    def __init__(self, max_iters: int = 20, tol_auc: float = 0.55, C: float = 1.0,
                 random_state: int = 0):
        self.max_iters = max_iters
        self.tol_auc = tol_auc
        self.C = C
        self.random_state = random_state

    def fit(self, X, y) -> "InlpEraser":
        """Fit on a single concept (``y`` shape ``(n,)``) or several at once
        (``y`` shape ``(n, k)`` -- e.g. LLM-expanded sub-concepts): each round projects out
        every still-recoverable concept column until none beats ``tol_auc``."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        X = _as_2d(X)
        Y = np.asarray(y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        cols = [_binarize(Y[:, j]) for j in range(Y.shape[1])]
        d = X.shape[1]
        P = np.eye(d)
        self.n_iters_ = 0
        for _ in range(self.max_iters):
            Xc = X @ P.T
            removed = False
            for yb in cols:
                if np.unique(yb).size < 2:
                    continue
                clf = LogisticRegression(max_iter=2000, C=self.C,
                                         random_state=self.random_state).fit(Xc, yb)
                try:
                    auc = roc_auc_score(yb, clf.predict_proba(Xc)[:, 1])
                except ValueError:
                    continue
                if auc <= self.tol_auc:
                    continue
                w = clf.coef_[0]
                n = np.linalg.norm(w)
                if n < 1e-12:
                    continue
                u = (w / n).reshape(-1, 1)
                P = (np.eye(d) - u @ u.T) @ P   # accumulate rank-1 nullspace projections
                Xc = X @ P.T
                removed = True
            if not removed:
                break
            self.n_iters_ += 1
        self.projection_ = P
        return self

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "projection_"):
            raise RuntimeError("call fit before transform")
        return _as_2d(X) @ self.projection_.T

    def fit_transform(self, X, y) -> np.ndarray:
        return self.fit(X, y).transform(X)


class LeaceEraser:
    """LEAst-squares Concept Erasure (closed form).

    Learns an affine map ``r(x) = x - A (x - mean)`` such that the cross-covariance
    between ``r(X)`` and the concept ``z`` is zero, while minimizing the expected
    squared change to ``x``. Accepts continuous or binary concept labels; multi-column
    ``z`` (e.g. one-hot) is supported.

    Parameters
    ----------
    shrinkage : ridge added to the feature covariance for numerical stability.
    """

    def __init__(self, shrinkage: float = 1e-3):
        self.shrinkage = shrinkage

    def fit(self, X, z) -> "LeaceEraser":
        X = _as_2d(X)
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        zc = z - z.mean(axis=0)

        Sxx = (Xc.T @ Xc) / n + self.shrinkage * np.eye(d)
        Sxz = (Xc.T @ zc) / n                      # (d, k) cross-covariance

        # Whitening W = Sxx^{-1/2} and its inverse Sxx^{1/2} via symmetric eigendecomp.
        vals, vecs = np.linalg.eigh(Sxx)
        vals = np.clip(vals, 1e-12, None)
        W = vecs @ np.diag(vals ** -0.5) @ vecs.T
        W_inv = vecs @ np.diag(vals ** 0.5) @ vecs.T

        M = W @ Sxz                                # concept directions in whitened space
        if M.size and np.linalg.norm(M) > 0:
            U, s, _ = np.linalg.svd(M, full_matrices=False)
            rank = int((s > 1e-8 * s.max()).sum())
            U = U[:, :rank]
            Pm = U @ U.T                           # projection onto span of concept directions
        else:
            Pm = np.zeros((d, d))
        self.proj_ = W_inv @ Pm @ W                # A: the affine "erase" operator
        return self

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "proj_"):
            raise RuntimeError("call fit before transform")
        Xc = _as_2d(X) - self.mean_
        return self.mean_ + Xc - Xc @ self.proj_.T

    def fit_transform(self, X, z) -> np.ndarray:
        return self.fit(X, z).transform(X)
