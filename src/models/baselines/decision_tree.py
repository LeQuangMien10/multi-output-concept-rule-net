"""
decision_tree.py — Minimal CART classifier, implemented from scratch in
NumPy (no scikit-learn — unavailable in this environment: PyPI downloads
kept resetting mid-transfer on this network).

Included as a classic, well-established rule-based interpretable-ML
baseline: every root-to-leaf path is literally an if/then rule, so it gives
an "off the shelf symbolic rule induction" reference point alongside the
neural ones (ICRL, CRL/RRL logic layers). Standard greedy, Gini-impurity,
pre-pruned by max_depth/min_samples_leaf — the textbook algorithm, not a
research contribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Node:
    is_leaf: bool = True
    predicted_class: int = 0
    class_counts: np.ndarray = None
    feature: int = -1
    threshold: float = 0.0
    left: "_Node" = None
    right: "_Node" = None
    depth: int = 0


def _gini(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - float((p ** 2).sum())


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 6, min_samples_leaf: int = 5, min_samples_split: int = 10):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.root: _Node | None = None
        self.num_classes = 0
        self.n_leaves = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        self.num_classes = int(y.max()) + 1
        self.n_leaves = 0
        self.root = self._build(X, y, depth=0)
        return self

    def _class_counts(self, y: np.ndarray) -> np.ndarray:
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for c in range(self.num_classes):
            counts[c] = int((y == c).sum())
        return counts

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        counts = self._class_counts(y)
        node = _Node(class_counts=counts, predicted_class=int(counts.argmax()), depth=depth)

        pure = (counts > 0).sum() <= 1
        if pure or depth >= self.max_depth or len(y) < self.min_samples_split:
            self.n_leaves += 1
            return node

        best = self._best_split(X, y, counts)
        if best is None:
            self.n_leaves += 1
            return node

        feature, threshold, left_mask = best
        node.is_leaf = False
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray, parent_counts: np.ndarray):
        n, d = X.shape
        parent_impurity = _gini(parent_counts)
        best_gain, best = 1e-12, None

        for feat in range(d):
            col = X[:, feat]
            order = np.argsort(col)
            col_sorted = col[order]
            y_sorted = y[order]

            left_counts = np.zeros(self.num_classes, dtype=np.int64)
            right_counts = parent_counts.copy()

            for i in range(n - 1):
                c = y_sorted[i]
                left_counts[c] += 1
                right_counts[c] -= 1
                if col_sorted[i] == col_sorted[i + 1]:
                    continue  # only split on genuine value boundaries
                n_left, n_right = i + 1, n - i - 1
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                impurity = (n_left * _gini(left_counts) + n_right * _gini(right_counts)) / n
                gain = parent_impurity - impurity
                if gain > best_gain:
                    threshold = (col_sorted[i] + col_sorted[i + 1]) / 2.0
                    best_gain = gain
                    best = (feat, threshold)

        if best is None:
            return None
        feature, threshold = best
        left_mask = X[:, feature] <= threshold
        if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
            return None
        return feature, threshold, left_mask

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x: np.ndarray, node: _Node) -> int:
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.predicted_class

    # ── Rule extraction: 1 leaf = 1 rule (conjunction of threshold conditions) ──

    def extract_rules(self, feature_names: list[str]) -> list[dict]:
        rules = []

        def walk(node: _Node, path: list[str]):
            if node.is_leaf:
                total = int(node.class_counts.sum())
                purity = float(node.class_counts.max() / total) if total else 0.0
                rules.append({
                    "conditions": list(path),
                    "predicts_class": node.predicted_class,
                    "n": total,
                    "purity": purity,
                    "class_counts": node.class_counts.tolist(),
                })
                return
            name = feature_names[node.feature]
            walk(node.left, path + [f"{name} <= {node.threshold:.3f}"])
            walk(node.right, path + [f"{name} > {node.threshold:.3f}"])

        walk(self.root, [])
        return rules

    def depth(self) -> int:
        def _d(node: _Node) -> int:
            if node.is_leaf:
                return node.depth
            return max(_d(node.left), _d(node.right))
        return _d(self.root)
