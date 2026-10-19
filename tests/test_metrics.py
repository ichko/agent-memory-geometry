from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agent_memory_geometry import metrics as M
from agent_memory_geometry import evaluation as E
from agent_memory_geometry.features import main, validate_inputs
from agent_memory_geometry.selection import select_indices


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.turns = np.array([[1, 0], [2, 0.2], [0, 1], [0.2, 2]], dtype=np.float32)
        self.memories = np.eye(2, dtype=np.float32)
        self.sessions = np.array([0, 0, 1, 1])
        self.query = np.array([1, 0], dtype=np.float32)

    def test_cosine_uses_original_space_and_normalizes(self):
        np.testing.assert_allclose(M.cosine_distance_matrix(np.array([[2., 0.]]), self.memories), [[0, 1]])

    def test_percentile_excludes_diagonal(self):
        self.assertEqual(M.calibrated_eps(self.memories, 25), 1)

    def test_threshold_is_strict(self):
        self.assertEqual(M.coverage_at_eps(self.memories, self.memories, 0), 0)
        self.assertEqual(M.coverage_at_eps(self.memories, self.memories, .01), 1)

    def test_query_topk_clips_to_memory_count(self):
        self.assertAlmostEqual(M.query_topk_mean(self.query, self.memories, k=5), .5)

    def test_rank_and_singletons(self):
        self.assertAlmostEqual(M.effective_rank(self.memories), 2, places=5)
        self.assertTrue(np.isnan(M.effective_rank(self.memories[:1])))
        self.assertTrue(np.isnan(M.redundancy_at_eps(self.memories[:1], 1)))

    def test_full_feature_set(self):
        features = M.all_features(self.turns, self.memories, self.sessions, q=self.query)
        self.assertEqual(len(E.ALL_GEOMETRY), 25)
        self.assertEqual(len(set(E.ALL_GEOMETRY)), 25)
        self.assertTrue(set(E.ALL_GEOMETRY) <= set(features))
        self.assertEqual(features["mem_count"], 2)
        self.assertEqual(features["n_sessions"], 2)

    def test_input_validation(self):
        validate_inputs(self.turns, self.memories, self.sessions, self.query)
        with self.assertRaises(ValueError):
            validate_inputs(self.turns, self.memories, self.sessions[:1], self.query)
        with self.assertRaises(ValueError):
            validate_inputs(self.turns, self.memories, self.sessions, np.zeros(2))
        with self.assertRaises(ValueError):
            validate_inputs(self.turns, self.memories, self.sessions, self.query, k=0)

    def test_npz_cli_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path, output_path = Path(directory) / "input.npz", Path(directory) / "result.csv"
            np.savez(input_path, turns=self.turns, memories=self.memories, sessions=self.sessions, query=self.query)
            with patch("sys.argv", ["features", str(input_path), "--output", str(output_path)]):
                self.assertEqual(main(), 0)
            self.assertIn("q_nn_mem", output_path.read_text())

    def test_selection(self):
        memories = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
        self.assertEqual(select_indices(self.query, memories, 1, "closest_k").tolist(), [0])
        self.assertEqual(select_indices(self.query, memories, 1, "farthest_k").tolist(), [2])
        a = select_indices(self.query, memories, 2, "random_k", seed=42)
        np.testing.assert_array_equal(a, select_indices(self.query, memories, 2, "random_k", seed=42))
        self.assertEqual(len(set(a)), 2)
        np.testing.assert_array_equal(select_indices(self.query, memories, 5, "closest_k"), np.arange(3))


if __name__ == "__main__":
    unittest.main()
