from __future__ import annotations

import gzip
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

from tools.package_demo import fingerprint, package, redact_display
from tools.build_landing_preview import build_preview_data, render_preview


class DemoPackageTests(unittest.TestCase):
    def test_all_pages_share_a_local_theme_colored_redis_favicon(self):
        site = Path(__file__).resolve().parents[1] / "site"
        svg = ET.fromstring((site / "favicon.svg").read_text())
        ns = {"svg": "http://www.w3.org/2000/svg"}
        self.assertEqual(svg.attrib["viewBox"], "0 0 32 32")
        self.assertEqual(svg.find("svg:title", ns).text, "Redis")
        self.assertEqual(svg.find("svg:path", ns).attrib["fill"], "#b51e2b")
        self.assertEqual(svg.find("svg:rect", ns).attrib["fill"], "#faf8f5")
        self.assertFalse(svg.findall(".//svg:script", ns))
        for name in ("index.html", "atlas.html", "data-notice.html"):
            self.assertIn('<link rel="icon" type="image/svg+xml" href="./favicon.svg?v=1">', (site / name).read_text())

    def test_landing_graphic_uses_an_accessible_button_free_measured_preview(self):
        site = Path(__file__).resolve().parents[1] / "site"
        page = (site / "index.html").read_text()

        class Graphic(HTMLParser):
            def __init__(self):
                super().__init__()
                self.images = []
                self.figures = []

            def handle_starttag(self, tag, attrs):
                values = dict(attrs)
                if tag == "img":
                    self.images.append(values)
                if tag == "figure":
                    self.figures.append(values)

        graphic = Graphic()
        graphic.feed(page)
        self.assertEqual(len(graphic.images), 1)
        self.assertEqual(graphic.images[0]["src"], "./atlas-preview.svg")
        self.assertIn("measured UMAP positions", graphic.images[0]["alt"])
        self.assertLessEqual(int(graphic.images[0]["width"]), 280)
        self.assertEqual(graphic.figures[0]["class"], "hero-preview")
        self.assertNotIn('id="preview-controls"', page)
        self.assertNotIn("data-preview-layout", page)
        self.assertNotIn('id="preview-layout"', page)
        self.assertNotIn('id="preview-hint"', page)
        self.assertIn("LongMemEval sample", page)
        self.assertNotIn("Closest source turns highlighted", page)
        self.assertEqual(page.count('<span>Forecasting Retrieval-Bound Memory Failures,</span>'), 1)
        preview_script = (site / "landing-preview.js").read_text()
        self.assertIn('canvas.addEventListener("pointerenter", () => { hovering = true; pauseForInput(); draw(); })', preview_script)
        self.assertIn('canvas.addEventListener("pointerleave", resumeAfterHover)', preview_script)
        self.assertIn("hovering = false; last = 0", preview_script)
        svg = ET.fromstring((site / "atlas-preview.svg").read_text())
        self.assertEqual(svg.attrib["role"], "img")
        ns = {"svg": "http://www.w3.org/2000/svg"}
        provenance = json.loads(svg.find("svg:metadata", ns).text)
        self.assertEqual(provenance["projection"], "umap")
        self.assertEqual(provenance["dataset"], "LongMemEval-small")
        self.assertLessEqual(provenance["turnSamples"], 1500)
        self.assertNotIn("Schematic", page)
        self.assertFalse(svg.findall(".//svg:script", ns))

    def test_preview_preserves_coordinates_and_session_boundaries(self):
        atlas = {"title": "Fixture", "embedding": {"model": "fixture"}, "projections": {"umap": {}},
                 "points": [{"umap": [0, 1], "text": "Not exported"}, {"umap": [2, 3]}, {"umap": [4, 2]}, {"umap": [5, 1]}],
                 "sessions": [{"id": "c1", "turns": [{"p": 0}, {"p": 1}]}, {"id": "c2", "turns": [{"p": 2}, {"p": 3}]}],
                 "histories": [{"memories": {"m2": [3]}}]}
        rendered = render_preview(atlas, turn_limit=3, path_limit=2)
        self.assertEqual(rendered, render_preview(atlas, turn_limit=3, path_limit=2))
        self.assertNotIn("Not exported", rendered)
        svg = ET.fromstring(rendered)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        for point in svg.findall(".//svg:circle", ns):
            original = atlas["points"][int(point.attrib["data-point-id"])]["umap"]
            self.assertEqual(float(point.attrib["cx"]), original[0])
            self.assertEqual(float(point.attrib["cy"]), -original[1])
        paths = svg.findall(".//svg:polyline", ns)
        self.assertEqual([p.attrib["data-session-id"] for p in paths], ["c1", "c2"])
        self.assertEqual(paths[0].attrib["points"], "0.000000,-1.000000 2.000000,-3.000000")
        self.assertEqual(paths[1].attrib["points"], "4.000000,-2.000000 5.000000,-1.000000")

    def test_landing_data_preserves_real_axes_and_has_no_source_text(self):
        atlas = {"title": "Fixture", "embedding": {"model": "fixture"}, "projections3d": {"umap": {}},
                 "tsneLayouts": {"local": {}},
                 "points": [{"umap3d": [i, i + 1, i + 2], "tsne_local": [i, -i], "text": "Do not export"} for i in range(5)],
                 "sessions": [{"id": "c1", "turns": [{"p": 0}, {"p": 1}]}, {"id": "c2", "turns": [{"p": 2}, {"p": 3}]}],
                 "histories": [{"memories": {"m2": [4]}}]}
        result = build_preview_data(atlas, turn_limit=2, path_limit=2)
        self.assertNotIn("Do not export", json.dumps(result))
        self.assertEqual(result["paths"], [[0, 1], [2, 3]])
        self.assertEqual(set(result["layouts"]), {"umap3d"})
        self.assertEqual(result["layouts"]["umap3d"]["coordinates"], [atlas["points"][i]["umap3d"] for i in result["pointIds"]])
        self.assertEqual(result["layouts"]["umap3d"]["dimensions"], 3)
        atlas["points"][0]["umap3d"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "measured coordinates"):
            build_preview_data(atlas)

    def test_paper_page_only_loads_preview_and_local_navigation_resolves(self):
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
                if tag == "img" and "src" in values:
                    self.links.append(values["src"])
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
                self.assertEqual(len(parsed.scripts), 1)
                self.assertEqual(parsed.scripts[0]["src"], "./landing-preview.js?v=9")
                self.assertEqual(parsed.scripts[0]["type"], "module")
                self.assertIn("connect-src 'none'", (site / page).read_text())

    def test_memory_preview_links_use_checked_original_space_scores(self):
        secret_example = "sk-" + "x" * 25
        atlas = {"title": "Fixture", "embedding": {"dimensions": 3}, "projections3d": {"umap": {}},
                 "correspondence": {"fingerprint": "fixture"},
                 "points": [{"umap3d": [i, i + 1, i + 2], "text": "Private source-turn sentinel"} for i in range(4)],
                 "sessions": [{"id": "c1", "turns": [{"p": 0}, {"p": 1}]}, {"id": "c2", "turns": [{"p": 2}]}],
                 "histories": [{"sessions": [0, 1], "memories": {"m2": [3]}}]}
        atlas["points"][3]["text"] = f"Actual memory with example token {secret_example}"
        matrix = {"schemaVersion": 1, "fingerprint": "fixture", "history": 0, "metric": "cosine", "dimensions": 3,
                  "rows": [[0, 0, 0], [0, 1, 1], [1, 0, 2]], "systems": {"m2": {"memories": [3], "scores": [.1, .9, .8]}}}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            target = folder / "h1.json"
            target.write_text(json.dumps(matrix))
            result = build_preview_data(atlas, turn_limit=1, path_limit=1, correspondence_dir=folder)
            detail = result["memoryDetails"][0]
            self.assertIn("Actual memory", detail["text"])
            self.assertNotIn(secret_example, detail["text"])
            self.assertNotIn("Private source-turn sentinel", json.dumps(result))
            self.assertEqual([result["pointIds"][m["point"]] for m in detail["matches"]], [1, 2, 0])
            self.assertEqual([m["score"] for m in detail["matches"]], [.9, .8, .1])
            self.assertEqual(result["backgroundPaths"], [0])
            self.assertEqual(result["paths"][detail["matches"][1]["path"]], [result["pointIds"].index(2)])
            matrix["fingerprint"] = "stale"
            target.write_text(json.dumps(matrix))
            with self.assertRaisesRegex(ValueError, "similarities do not match"):
                build_preview_data(atlas, correspondence_dir=folder)
            matrix["fingerprint"] = "fixture"
            matrix["systems"]["m2"]["scores"][0] = 1.1
            target.write_text(json.dumps(matrix))
            with self.assertRaisesRegex(ValueError, "similarities do not match"):
                build_preview_data(atlas, correspondence_dir=folder)

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
            self.assertTrue((root / "site/atlas-preview.svg").exists())
            self.assertTrue((root / "site/preview-data.mjs").exists())
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
