from __future__ import annotations

import unittest

from agent_memory_geometry.diagnostics import anchor_bins, reachability_gaps


class DiagnosticsTests(unittest.TestCase):
    def test_proxy_bins_are_exclusive_and_thresholded_on_correct(self):
        rows = [dict(question_type="a", score=1, answer_distance=.2, query_distance=.2),
                dict(question_type="a", score=0, answer_distance=.9, query_distance=.9),
                dict(question_type="a", score=0, answer_distance=.1, query_distance=.9),
                dict(question_type="a", score=0, answer_distance=.1, query_distance=.1)]
        pooled = anchor_bins(rows)[-1]
        self.assertEqual(pooled["n_failed"], 3)
        self.assertEqual(pooled["answer_threshold"], .2)
        self.assertEqual(pooled["answer_distant"], 1)
        self.assertEqual(pooled["query_distant"], 1)
        self.assertEqual(pooled["neither"], 1)

    def test_gap_conditions_on_correctness(self):
        rows = [dict(question_type="a", score=1, q_nn_mem=.2),
                dict(question_type="a", score=0, q_nn_mem=.5)]
        self.assertAlmostEqual(reachability_gaps(rows)[0]["gap"], .3)

    def test_proxy_needs_correct_examples(self):
        with self.assertRaises(ValueError):
            anchor_bins([dict(question_type="a", score=0, answer_distance=.9, query_distance=.9)])


if __name__ == "__main__":
    unittest.main()
