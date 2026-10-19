from __future__ import annotations

import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from agent_memory_geometry import evaluation as E
from agent_memory_geometry.reproduce import paired_delta


class EvaluationTests(unittest.TestCase):
    def test_auroc_ties_and_reference(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        p = np.array([.1, .1, .2, .7, .9, .9])
        self.assertEqual(E._auroc(y, p), roc_auc_score(y, p))
        self.assertEqual(E._auroc(y, np.ones(len(y))), .5)
        self.assertTrue(np.isnan(E._auroc(np.ones(4), np.arange(4))))

    def test_historical_imputation(self):
        X = E._matrix([{"a": "nan", "b": ""}, {"a": 4, "b": "nan"}], ["a", "b"])
        np.testing.assert_array_equal(X, [[4, 0], [4, 0]])

    def test_grouped_oof_matches_manual_training(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 3))
        y = np.tile([0, 1], 60)
        groups = np.repeat(np.arange(40), 3)
        actual = E._grouped_cv_oof(X, y, groups)
        expected = np.full(len(y), np.nan)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups):
            self.assertFalse(set(groups[tr]) & set(groups[te]))
            scaler = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=1, solver="lbfgs", max_iter=2000).fit(scaler.transform(X[tr]), y[tr])
            expected[te] = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        np.testing.assert_allclose(actual, expected)

    def test_insufficient_classes(self):
        self.assertIsNone(E._grouped_cv_oof(np.ones((10, 2)), np.ones(10), np.arange(10)))

    def test_paired_comparison_identical(self):
        y = np.tile([0, 1], 20)
        p = np.arange(len(y)) / len(y)
        result = paired_delta(y, p, p, np.repeat(np.arange(20), 2), n_boot=30)
        self.assertEqual(result, {"delta": 0, "ci_lo": 0, "ci_hi": 0})


if __name__ == "__main__":
    unittest.main()
