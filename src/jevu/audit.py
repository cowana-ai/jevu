"""Auditing helpers: how linearly recoverable is a concept before vs. after erasure."""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["concept_auc", "erasure_report", "tpr_gap"]


def concept_auc(X, y, test_size: float = 0.3, seed: int = 0, C: float = 1.0) -> float:
    """ROC-AUC of a linear probe predicting concept ``y`` from embeddings ``X``.

    1.0 = perfectly linearly recoverable, ~0.5 = erased. Uses a train/test split so the
    number reflects generalization, not memorization.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X = np.asarray(X, dtype=float)
    yb = (np.asarray(y) >= 0.5).astype(int) if np.asarray(y).dtype.kind == "f" else np.asarray(y).astype(int)
    if np.unique(yb).size < 2:
        return float("nan")
    Xtr, Xte, ytr, yte = train_test_split(X, yb, test_size=test_size, random_state=seed, stratify=yb)
    clf = LogisticRegression(max_iter=2000, C=C, random_state=seed).fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def erasure_report(X_before, X_after, y, test_size: float = 0.3, seed: int = 0) -> dict:
    """Concept recoverability before vs. after erasure, plus the drop."""
    before = concept_auc(X_before, y, test_size, seed)
    after = concept_auc(X_after, y, test_size, seed)
    above_chance = max(before - 0.5, 1e-9)
    return {
        "concept_auc_before": before,
        "concept_auc_after": after,
        "fraction_removed": float((before - after) / above_chance),
    }


def tpr_gap(true_label, pred_label, group, min_cell: int = 10) -> float:
    """Mean absolute true-positive-rate gap between two groups across classes.

    A standard fairness metric (e.g. Bias in Bios): for each class ``c`` compares recall
    for ``group==1`` vs ``group==0``, averaging ``|TPR_1 - TPR_0|`` over classes that have
    at least ``min_cell`` examples in each group. Lower is fairer.
    """
    true_label = np.asarray(true_label)
    pred_label = np.asarray(pred_label)
    group = np.asarray(group)
    gaps = []
    for c in np.unique(true_label):
        a = (true_label == c) & (group == 1)
        b = (true_label == c) & (group == 0)
        if a.sum() >= min_cell and b.sum() >= min_cell:
            gaps.append(abs(np.mean(pred_label[a] == c) - np.mean(pred_label[b] == c)))
    return float(np.mean(gaps)) if gaps else float("nan")
