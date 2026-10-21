from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.build_atlas import collect, add_3d, add_tsne, export_correspondence, select_tsne_layout


class AtlasExportTests(unittest.TestCase):
    def rows(self):
        return [
            {"question_id": "a", "question": "First question", "haystack_sessions": [[{"role": "user", "content": "First turn"}, {"role": "assistant", "content": "Second turn"}]], "haystack_dates": ["date"]},
            {"question_id": "b", "question": "Second question", "haystack_sessions": [[{"role": "user", "content": "First turn"}, {"role": "assistant", "content": "Second turn"}], [{"role": "user", "content": "Third turn"}]]},
        ]

    def test_deduplicate_sessions_without_merging_paths(self):
        result = collect(self.rows(), [])
        self.assertEqual(len(result["histories"]), 2)
        self.assertEqual(len(result["sessions"]), 2)
        self.assertEqual(result["sessions"][0]["histories"], [0, 1])
        self.assertEqual(result["histories"][1]["sessions"], [0, 1])
        self.assertEqual(len(result["sessions"][0]["turns"]), 2)

    def test_memory_sampling_keeps_full_counts_and_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answers.jsonl"
            path.write_text(json.dumps({"question_id": "a", "memories": [f"Memory {i}" for i in range(100)]}) + "\n")
            result = collect(self.rows(), [("Test", path)], memory_limit=4)
        history = result["histories"][0]
        self.assertEqual(history["memoryTotals"]["m1"], 100)
        self.assertEqual(len(history["memories"]["m1"]), 4)
        self.assertEqual(result["texts"][history["memories"]["m1"][0]], "Memory 0")
        self.assertEqual(result["texts"][history["memories"]["m1"][-1]], "Memory 99")

    def test_text_and_vectors_are_not_fabricated(self):
        result = collect(self.rows(), [])
        self.assertEqual(result["embedding"]["model"], "redis/langcache-embed-v3-small")
        self.assertEqual(result["embedding"]["kind"], "analysis")
        self.assertNotIn("points", result)
        self.assertIn("First turn", result["texts"])

    def test_adding_3d_preserves_existing_2d_coordinates_and_text(self):
        class PCAFixture:
            explained_variance_ratio_ = np.array([.4, .3, .2])

            def __init__(self, **kwargs):
                pass

            def fit_transform(self, vectors):
                return np.asarray(vectors[:, :3])

        class UMAPFixture:
            def __init__(self, **kwargs):
                if kwargs["n_components"] != 3:
                    raise AssertionError("The extra UMAP fit must be genuinely three-dimensional")

            def fit_transform(self, vectors):
                return vectors + .25

        original = [{"text": f"Unit test {i}", "pca": [i, i + 1], "umap": [i + 2, i + 3]} for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "atlas.json").write_text(json.dumps({"points": original}))
            np.save(output / "vectors.npy", np.arange(16, dtype=np.float32).reshape(4, 4))
            with patch.dict("sys.modules", {"umap": types.SimpleNamespace(UMAP=UMAPFixture), "sklearn.decomposition": types.SimpleNamespace(PCA=PCAFixture)}):
                add_3d(output)
            result = json.loads((output / "atlas.json").read_text())
        for before, after in zip(original, result["points"]):
            self.assertEqual({key: after[key] for key in before}, before)
            self.assertEqual(len(after["pca3d"]), 3)
            self.assertEqual(len(after["umap3d"]), 3)
        self.assertEqual(result["projections3d"]["umap"]["dimensions"], 3)

    def test_tsne_preserves_points_and_existing_layouts(self):
        class PCAFixture:
            def __init__(self, **kwargs):
                pass

            def fit_transform(self, vectors):
                return vectors[:, :3]

        class TSNEFixture:
            def __init__(self, **kwargs):
                self.settings = kwargs
                if kwargs["metric"] != "euclidean" or kwargs["random_state"] != 42 or kwargs["n_components"] != 2:
                    raise AssertionError("Shared t-SNE protocol must be recorded and explicit")

            def fit(self, vectors):
                return vectors[:, :2] + .5

        original = [{"text": f"Unit test {i}", "pca": [i, i + 1], "umap": [i + 2, i + 3], "pca3d": [i, i + 1, i + 2]} for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "atlas.json").write_text(json.dumps({"points": original, "correspondence": {"fingerprint": "preserve"}}))
            np.save(output / "vectors.npy", np.arange(16, dtype=np.float32).reshape(4, 4))
            with patch.dict("sys.modules", {"openTSNE": types.SimpleNamespace(TSNE=TSNEFixture), "sklearn.decomposition": types.SimpleNamespace(PCA=PCAFixture)}):
                add_tsne(output, perplexity=1)
            result = json.loads((output / "atlas.json").read_text())
        for before, after in zip(original, result["points"]):
            self.assertEqual({key: after[key] for key in before}, before)
            self.assertEqual(len(after["tsne"]), 2)
        self.assertEqual(result["correspondence"]["fingerprint"], "preserve")
        self.assertEqual(result["projections"]["tsne"]["fitPoints"], 4)

    def test_correspondence_uses_original_vectors_not_projection_distances(self):
        atlas = {"embedding": {"model": "test-only", "dimensions": 3}, "points": [{"text": str(i), "umap": [0, 0]} for i in range(3)], "sessions": [{"turns": [{"p": 0}, {"p": 1}]}], "histories": [{"sessions": [0], "memories": {"m1": [2]}, "memoryTotals": {"m1": 100}}]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "atlas.json").write_text(json.dumps(atlas))
            np.save(output / "vectors.npy", np.array([[2., 0, 0], [0, 1., 0], [1., 0, 0]], dtype=np.float32))
            export_correspondence(output)
            result = json.loads((output / "correspondence/h1.json").read_text())
            updated = json.loads((output / "atlas.json").read_text())
        self.assertEqual(result["rows"], [[0, 0, 0], [0, 1, 1]])
        self.assertEqual(result["systems"]["m1"]["scores"], [1., 0.])
        self.assertEqual(result["systems"]["m1"]["total"], 100)
        self.assertEqual(result["fingerprint"], updated["correspondence"]["fingerprint"])
        self.assertEqual(updated["points"], atlas["points"])

    def test_named_tsne_fits_preserve_original_and_previous_variants(self):
        class PCAFixture:
            def __init__(self, **kwargs):
                pass

            def fit_transform(self, vectors):
                return vectors[:, :3]

        class TSNEFixture:
            def __init__(self, **kwargs):
                self.offset = kwargs["perplexity"]

            def fit(self, vectors):
                if vectors.shape[1] == 4 and not vectors.flags.writeable:
                    raise AssertionError("NN-descent requires writable arrays, including raw cached vectors")
                return vectors[:, :2] + self.offset

        points = [{"text": str(i), "tsne": [i, -i], "umap": [i, i], "pca": [0, i]} for i in range(7)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "atlas.json").write_text(json.dumps({"points": points, "projections": {"tsne": {"perplexity": 30}}, "correspondence": {"fingerprint": "keep"}}))
            np.save(output / "vectors.npy", np.arange(28, dtype=np.float32).reshape(7, 4))
            with patch.dict("sys.modules", {"openTSNE": types.SimpleNamespace(TSNE=TSNEFixture), "sklearn.decomposition": types.SimpleNamespace(PCA=PCAFixture)}):
                add_tsne(output, perplexity=1, layout="local")
                add_tsne(output, perplexity=2, layout="broad", metric="cosine", exaggeration=1)
                add_tsne(output, perplexity=1, layout="full", metric="cosine", pca_dimensions=0)
            select_tsne_layout(output, "full")
            result = json.loads((output / "atlas.json").read_text())
        self.assertEqual(result["tsneDefault"], "full")
        self.assertEqual(result["correspondence"]["fingerprint"], "keep")
        self.assertEqual(set(result["tsneLayouts"]), {"original", "local", "broad", "full"})
        self.assertEqual(result["tsneLayouts"]["broad"]["metric"], "cosine")
        self.assertEqual(result["tsneLayouts"]["local"]["iterations"], 1000)
        self.assertEqual(result["tsneLayouts"]["full"]["pcaDimensions"], 0)
        for before, after in zip(points, result["points"]):
            self.assertEqual({key: after[key] for key in before}, before)
            self.assertNotEqual(after["tsne_local"], after["tsne_broad"])

    def test_correspondence_rejects_zero_or_mismatched_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "atlas.json").write_text(json.dumps({"points": [{"text": "Unit test"}], "embedding": {}}))
            np.save(output / "vectors.npy", np.zeros((1, 3), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "zero embedding"):
                export_correspondence(output)
            np.save(output / "vectors.npy", np.ones((2, 3), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "match atlas"):
                export_correspondence(output)


if __name__ == "__main__":
    unittest.main()
