from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

from tools.package_demo import fingerprint, package, redact_display


class DemoPackageTests(unittest.TestCase):
    def test_paper_page_is_script_free_and_local_navigation_resolves(self):
        site = Path(__file__).resolve().parents[1] / "site"

        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links = []
                self.scripts = []

            def handle_starttag(self, tag, attrs):
                values = dict(attrs)
                if tag == "a" and "href" in values:
                    self.links.append(values["href"])
                if tag == "script":
                    self.scripts.append(values)

        for page in ["index.html", "atlas.html", "data-notice.html"]:
            parsed = Links()
            parsed.feed((site / page).read_text())
            for link in parsed.links:
                url = urlsplit(link)
                if not url.scheme and not url.netloc:
                    target = site / (url.path or page)
                    self.assertTrue(target.is_file() or (target.is_dir() and (target / "index.html").is_file()), f"Broken local link: {page}: {link}")
            if page == "index.html":
                self.assertFalse(parsed.scripts)
                self.assertIn("connect-src 'none'", (site / page).read_text())

    def fixture(self, root):
        texts = ["Source turn", "Question", "Memory"]
        points = [{"text": text, "umap": [i, i], "pca": [i, i]} for i, text in enumerate(texts)]
        atlas = {"schemaVersion": 2, "title": "Unit test", "source": "https://example.org/public", "embedding": {"model": "test-only", "kind": "analysis", "dimensions": 3}, "points": points, "systems": [{"id": "m1", "label": "Test"}], "sessions": [{"id": "c1", "label": texts[0], "turns": [{"p": 0, "role": "user"}], "histories": [0]}], "histories": [{"id": "h1", "question": texts[1], "query": 1, "sessions": [0], "memories": {"m1": [2]}, "memoryTotals": {"m1": 1}}]}
        atlas["correspondence"] = {"fingerprint": fingerprint(atlas)}
        (root / "atlas.json").write_text(json.dumps(atlas))
        (root / "public.json").write_text(json.dumps([{"question": texts[1], "haystack_sessions": [[{"content": texts[0]}]]}]))
        (root / "correspondence").mkdir()
        (root / "correspondence/h1.json").write_text(json.dumps({"schemaVersion": 1, "fingerprint": atlas["correspondence"]["fingerprint"], "history": 0, "metric": "cosine", "dimensions": 3, "rows": [[0, 0, 0]], "systems": {"m1": {"memories": [2], "total": 1, "scores": [.75]}}}))
        return atlas

    def test_public_pack_preserves_coordinates_scores_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atlas = self.fixture(root)
            result = package(root / "atlas.json", root / "public.json", root / "site", chunk_size=2)
            manifest = json.loads((root / "site/atlas.json").read_text())
            points = [point for part in manifest["parts"] for point in json.loads(gzip.decompress((root / "site" / part).read_bytes()))]
            self.assertEqual(points, atlas["points"])
            self.assertEqual(json.loads(gzip.decompress((root / "site/correspondence/h1.json.gz").read_bytes()))["systems"]["m1"]["scores"], [.75])
            self.assertEqual(result["histories"], 1)
            self.assertTrue((root / "site/.nojekyll").exists())
            landing = (root / "site/index.html").read_text()
            self.assertIn('href="./atlas.html"', landing)
            self.assertNotIn("app.js", landing)
            self.assertIn('id="atlas-canvas"', (root / "site/atlas.html").read_text())
            self.assertTrue((root / "site/landing.css").exists())
            self.assertFalse((root / "site/vectors.npy").exists())
            with self.assertRaisesRegex(ValueError, "empty output"):
                package(root / "atlas.json", root / "public.json", root / "site")

    def test_private_source_turn_or_unexpected_fields_block_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atlas = self.fixture(root)
            atlas["points"][0]["text"] = "Not a public benchmark turn"
            (root / "atlas.json").write_text(json.dumps(atlas))
            with self.assertRaisesRegex(ValueError, "absent from the public"):
                package(root / "atlas.json", root / "public.json", root / "site")
            atlas["points"][0]["backend_config"] = {}
            (root / "atlas.json").write_text(json.dumps(atlas))
            with self.assertRaisesRegex(ValueError, "Unexpected atlas fields"):
                package(root / "atlas.json", root / "public.json", root / "site")

    def test_display_redaction_and_private_source_guard(self):
        token = "sk-" + "a" * 40
        self.assertNotIn(token, redact_display("An example: " + token))
        self.assertEqual(redact_display("Memory of a public conversation"), "Memory of a public conversation")
        self.assertNotIn("user:pass", redact_display("redis" + "://user:pass@localhost:6379"))
        with self.assertRaisesRegex(ValueError, "Private-source"):
            redact_display("redis" + "labsdev")


if __name__ == "__main__":
    unittest.main()
