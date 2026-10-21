"""Package an explicitly approved benchmark atlas as a standalone static site."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{20,})\b")
KEY = re.compile(r"\x2d{5}BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY\x2d{5}.*?(?:\x2d{5}END (?:RSA |EC |OPENSSH )?PRIVATE KEY\x2d{5}|\Z)", re.S)
CONNECTION = re.compile(r"(?:redis|rediss|postgres(?:ql)?|mongodb(?:\+srv)?):\/\/[^\s`\"'<>]+", re.I)
LOCAL_PATH = re.compile(r"/(?:Users|private|home)/[A-Za-z0-9_.-]+/[^\s`\"'<>]*")
POINT_KEYS = {"text", "umap", "pca", "umap3d", "pca3d", "tsne", "tsne_local", "tsne_broad", "tsne_full"}
TOP_KEYS = {"schemaVersion", "title", "source", "embedding", "systems", "histories", "sessions", "points", "projections", "projections3d", "tsneLayouts", "tsneDefault", "correspondence"}


def redact_display(text: str) -> str:
    """Remove credential-shaped examples without changing any plotted coordinate."""
    lowered = text.lower()
    if any(marker in lowered for marker in ("redis" + "labsdev", "ai-research-" + "semantic-cache", ".redisenterprise" + ".com", ".redislabs" + ".com")):
        raise ValueError("Private-source marker found; refusing to package the demo")
    for pattern, replacement in ((KEY, "[private-key example omitted]"), (TOKEN, "[credential example omitted]"), (CONNECTION, "[connection example omitted]"), (LOCAL_PATH, "[local path omitted]")):
        text = pattern.sub(replacement, text)
    return text


def fingerprint(atlas: dict) -> str:
    digest = hashlib.sha256(json.dumps(atlas["embedding"], sort_keys=True).encode())
    for point in atlas["points"]:
        text = point["text"].encode()
        digest.update(len(text).to_bytes(8, "little"))
        digest.update(text)
    return digest.hexdigest()


def write_gzip(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
    if path.stat().st_size >= 25 * 1024 * 1024:
        raise ValueError(f"Static data part exceeds the 25 MiB packaging limit: {path.name}")


def package(atlas_path: Path, dataset_path: Path, output: Path, chunk_size: int = 10000) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory; existing files are never overwritten")
    if not 1 <= chunk_size <= 20000:
        raise ValueError("Point chunk size must be between 1 and 20000")
    atlas = json.loads(atlas_path.read_text())
    if set(atlas) - TOP_KEYS or any(set(point) - POINT_KEYS for point in atlas["points"]):
        raise ValueError("Unexpected atlas fields; review them before publication")
    public = json.loads(dataset_path.read_text())
    source_texts = {hashlib.sha256(str(turn.get("content", "")).strip().encode()).digest() for history in public for session in history["haystack_sessions"] for turn in session}
    if len(public) != len(atlas["histories"]) or any(history["question"] != row["question"] for history, row in zip(atlas["histories"], public)):
        raise ValueError("Atlas questions do not match the supplied public benchmark")
    for session in atlas["sessions"]:
        for turn in session["turns"]:
            if hashlib.sha256(atlas["points"][turn["p"]]["text"].encode()).digest() not in source_texts:
                raise ValueError("A conversation turn is absent from the public benchmark")
    original_fingerprint = fingerprint(atlas)
    if atlas.get("correspondence", {}).get("fingerprint") != original_fingerprint:
        raise ValueError("Atlas correspondence fingerprint is stale")
    changed = 0
    for point in atlas["points"]:
        clean = redact_display(point["text"])
        changed += clean != point["text"]
        point["text"] = clean
    for session in atlas["sessions"]:
        session["label"] = redact_display(session["label"])
    for history in atlas["histories"]:
        history["question"] = redact_display(history["question"])
    public_fingerprint = fingerprint(atlas)
    atlas["correspondence"]["fingerprint"] = public_fingerprint
    atlas["correspondence"]["files"] = "correspondence/h{history}.json.gz"
    atlas["publication"] = {
        "dataset": atlas["source"], "datasetLicense": "MIT", "redactedPoints": changed,
        "note": "Credential-shaped examples and local paths are omitted from displayed text. Coordinates and similarities are unchanged and were computed before display redaction. Memory overlays are sampled experimental outputs, not the full stores.",
    }
    output.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "site").iterdir():
        if source.is_file() and (source.suffix in {".html", ".css", ".js", ".mjs"} or source.name == "atlas-preview.svg"):
            shutil.copyfile(source, output / source.name)
    (output / ".nojekyll").write_text("")
    points = atlas.pop("points")
    write_gzip(output / "data/metadata.json.gz", atlas)
    parts = []
    for start in range(0, len(points), chunk_size):
        path = f"data/points-{start // chunk_size:03d}.json.gz"
        write_gzip(output / path, points[start:start + chunk_size])
        parts.append(path)
        print(f"Packed points {min(start + chunk_size, len(points)):,}/{len(points):,}", flush=True)
    (output / "atlas.json").write_text(json.dumps({"format": "memory-atlas-pack-v1", "metadata": "data/metadata.json.gz", "parts": parts, "pointCount": len(points)}, indent=2) + "\n")
    for index, history in enumerate(atlas["histories"]):
        matrix = json.loads((atlas_path.parent / "correspondence" / f"h{index + 1}.json").read_text())
        rows = [[session, turn, item["p"]] for session in history["sessions"] for turn, item in enumerate(atlas["sessions"][session]["turns"])]
        if set(matrix) != {"schemaVersion", "fingerprint", "history", "metric", "dimensions", "rows", "systems"} or matrix["fingerprint"] != original_fingerprint or matrix["history"] != index or matrix["rows"] != rows:
            raise ValueError("Similarity sidecar does not match the atlas")
        for system, memories in history["memories"].items():
            block = matrix["systems"][system]
            if set(block) != {"memories", "total", "scores"} or block["memories"] != memories or len(block["scores"]) != len(rows) * len(memories):
                raise ValueError("Unexpected similarity columns or fields")
        matrix["fingerprint"] = public_fingerprint
        write_gzip(output / f"correspondence/h{index + 1}.json.gz", matrix)
        if (index + 1) % 50 == 0:
            print(f"Packed similarities {index + 1}/{len(atlas['histories'])}", flush=True)
    files = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.rglob("*")) if p.is_file()]
    manifest = {"schemaVersion": 1, "histories": len(atlas["histories"]), "conversations": len(atlas["sessions"]), "points": len(points), "redactedPoints": changed, "sourceSha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(), "files": files}
    (output / "publication-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Demo ready: {len(files)} files, {sum(p['bytes'] for p in files) / 1e6:.1f} MB; {changed} display-text redactions", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve-public-data", action="store_true", help="Confirm explicit approval to publish benchmark conversations and sampled memories")
    args = parser.parse_args()
    if not args.approve_public_data:
        parser.error("Publication requires --approve-public-data after reviewing data permissions")
    package(args.atlas, args.dataset, args.output)


if __name__ == "__main__":
    main()
