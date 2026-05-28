"""
Ordinal regression classifier via cumulative threshold approach.

Extracted as a shared module so it can be imported by both
train_affect_risk.py (training) and api.py (inference via joblib.load).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class OrdinalClassifier:
    """
    Ordinal regression via cumulative threshold approach.

    Instead of treating ordinal labels as unrelated categories,
    this fits K-1 binary classifiers for P(Y > k) and combines
    them to produce ordinal predictions.

    For 3 ordinal levels (1, 2, 3):
      - clf_1: P(Y > 1) — separates level 1 from {2, 3}
      - clf_2: P(Y > 2) — separates {1, 2} from level 3

    Final prediction uses cumulative probabilities:
      P(Y = 1) = 1 - P(Y > 1)
      P(Y = 2) = P(Y > 1) - P(Y > 2)
      P(Y = 3) = P(Y > 2)
    """

    def __init__(self, base_clf=None):
        self.base_clf = base_clf or LogisticRegression(
            max_iter=1000, solver='lbfgs', C=1.0, random_state=42
        )
        self.clfs = {}
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.sort(np.unique(y))
        # Fit K-1 binary classifiers
        for k in self.classes_[:-1]:
            binary_y = (y > k).astype(int)
            clf = LogisticRegression(
                max_iter=1000, solver='lbfgs', C=1.0, random_state=42
            )
            # Only fit if both classes exist
            if len(np.unique(binary_y)) > 1:
                clf.fit(X, binary_y, sample_weight=sample_weight)
            else:
                # Degenerate case — always predict the single class
                clf = None
            self.clfs[k] = clf
        return self

    def predict_proba(self, X):
        """Return probability for each ordinal level."""
        n = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n, n_classes))

        # Get cumulative probabilities P(Y > k)
        cum_probs = {}
        for k in self.classes_[:-1]:
            clf = self.clfs[k]
            if clf is not None:
                cum_probs[k] = clf.predict_proba(X)[:, 1]
            else:
                cum_probs[k] = np.zeros(n)

        # Convert cumulative to category probabilities
        # P(Y = first_class) = 1 - P(Y > first_class)
        proba[:, 0] = 1 - cum_probs[self.classes_[0]]
        # P(Y = last_class) = P(Y > second_to_last)
        proba[:, -1] = cum_probs[self.classes_[-2]]
        # P(Y = middle_k) = P(Y > k-1) - P(Y > k)
        for i in range(1, n_classes - 1):
            prev = cum_probs[self.classes_[i - 1]]
            curr = cum_probs[self.classes_[i]]
            proba[:, i] = prev - curr

        # Clip and normalize (numerical stability)
        proba = np.clip(proba, 0, 1)
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        proba = proba / row_sums

        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
