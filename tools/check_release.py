"""Check the publication allowlist, numeric schemas, and common leak markers.

This is a review aid, not a complete secret detector. It does not upload files.
PDF text and metadata need a separate review before publication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent_memory_geometry.evaluation import ALL_GEOMETRY

SYSTEMS = {"Sem-RAG", "Ext-Dense", "Ext-Hybrid", "Obs-Extract", "Svc-Mem"}
TYPES = {"single-session-assistant", "single-session-user", "single-session-preference",
         "multi-session", "temporal-reasoning", "knowledge-update"}
FIELDS = {
    "geometry": ["question_id", "question_type", "strategy", "score", *ALL_GEOMETRY],
    "question_covariates": ["question_id", "strategy", "q_len_chars", *[f"qpca64_{j}" for j in range(64)]],
    "confidence": ["strategy", "question_id", "confidence", "p_true"],
    "answer_anchors": ["question_id", "question_type", "strategy", "score", "answer_distance", "query_distance"],
    "selection_longmemeval": ["question_id", "question_type", "condition", "n_memories_used", "score", "k", "answer_model"],
    "selection_locomo": ["conversation_id", "question_id", "condition", "n_memories_used", "score", "k"],
    "nearest_turn": ["conversation_id", "question_id", "category", "score", "nearest_turn_distance"],
}
PUBLIC_EMAILS = {"iliya.zhechev@gmail.com", "radoslav.ralev@redis.com", "ccol@u.northwestern.edu",
                 "srijith.rajamohan@gmail.com", "honey.agarwal@gmail.com",
                 "tyler.hutcherson@redis.com", "simba.khadder@redis.com", "garnett@wustl.edu"}
PATTERNS = {
    "credential token": r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{20,})\b",
    "private key": r"\x2d{5}BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY",
    "service connection string": r"(?:redis|rediss|postgres(?:ql)?|mongodb(?:\+srv)?):\/\/",
    "local absolute path": r"/(?:Users|private|home)/[A-Za-z0-9_.-]+/",
    "private source organization": r"[r]edislabsdev",
    "private source workspace": r"[a]i-research-semantic-cache",
    "historical run identifier": r"\b20\d{6}-\d{4}-[a-z]",
    "private service endpoint": r"\b[a-zA-Z0-9.-]+\.(?:redisenterprise|redislabs)\.com\b",
}


def inspect_text(text, filename):
    problems = []
    for name, pattern in PATTERNS.items():
        if re.search(pattern, text, re.I):
            problems.append(f"{filename}: {name}")
    emails = set(re.findall(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text))
    if emails - PUBLIC_EMAILS:
        problems.append(f"{filename}: unapproved email address")
    return problems


def csv_manifest(path):
    relative = path.relative_to(ROOT).as_posix()
    if path.name.startswith("geometry_"):
        kind = "geometry"
    elif path.parent.name == "context_selection":
        kind = "selection_" + path.stem
    else:
        kind = path.stem
    if kind not in FIELDS:
        raise ValueError(f"Unexpected data file: {relative}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS[kind]:
            raise ValueError(f"Unexpected CSV fields: {relative}")
        rows = list(reader)
    for row in rows:
        for key, value in row.items():
            if key == "question_id":
                valid = bool(re.fullmatch(r"q\d{4}|c\d{2}_q\d{4}", value))
            elif key == "conversation_id":
                valid = bool(re.fullmatch(r"c\d{2}", value))
            elif key == "strategy":
                valid = value in SYSTEMS
            elif key == "question_type":
                valid = value in TYPES or (kind == "selection_longmemeval" and value == "")
            elif key == "condition":
                valid = bool(re.fullmatch(r"all|rich|closest_k|native_k|farthest_k|random_k(?:_s[1-4])?", value))
            elif key == "answer_model":
                valid = value in {"gpt-4o", "gpt-4o-mini"}
            elif key == "score":
                valid = value in {"", "-1", "0", "1"}
            else:
                try:
                    number = float(value)
                    valid = math.isfinite(number) or (kind in {"geometry", "confidence"} and value.lower() == "nan")
                except (ValueError, TypeError):
                    valid = False
            if not valid:
                raise ValueError(f"Non-allowlisted CSV value in {relative}, field {key}")
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size, "rows": len(rows), "fields": FIELDS[kind]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--text-file", type=Path, help="Also scan extracted PDF text or metadata")
    args = parser.parse_args()
    failures = []
    files = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(p in {".git", ".venv", "__pycache__", "outputs", "dist", ".pytest_cache"} for p in relative.parts):
            continue
        if path.is_symlink():
            failures.append(f"{relative}: symlink not permitted")
            continue
        if not path.is_file():
            continue
        allowed = (len(relative.parts) == 1 and path.name in {".gitattributes", ".gitignore", "AGENTS.md", "CITATION.cff", "NOTICE.md", "README.md", "pyproject.toml", "uv.lock"})
        allowed |= relative.parts[0] in {"src", "tests", "tools"} and path.suffix == ".py"
        allowed |= relative.as_posix() == "tests/explorer.test.mjs"
        allowed |= relative.parts[0] == "site" and path.suffix in {".html", ".css", ".js", ".mjs"}
        allowed |= relative.as_posix() in {"site/atlas-preview.svg", "site/favicon.svg"}
        allowed |= relative.as_posix() == ".github/workflows/pages.yml"
        allowed |= relative.parts[0] == "docs" and path.suffix == ".md"
        allowed |= relative.parts[0] == "data" and (path.suffix == ".csv" or relative.as_posix() in {"data/README.md", "data/manifest.json"})
        allowed |= relative.parts[0] == "paper" and path.suffix in {".tex", ".bbl", ".bib", ".sty", ".pdf"}
        if not allowed:
            failures.append(f"{relative}: file outside publication allowlist")
        if path.suffix != ".pdf":
            failures.extend(inspect_text(path.read_text(), str(relative)))
        files.append(path)
    manifest = {"schema_version": 1, "artifacts": [csv_manifest(p) for p in sorted((ROOT / "data").rglob("*.csv"))]}
    if args.text_file:
        failures.extend(inspect_text(args.text_file.read_text(), "extracted PDF content"))
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        raise SystemExit(1)
    target = ROOT / "data/manifest.json"
    if args.write_manifest:
        target.write_text(json.dumps(manifest, indent=2) + "\n")
    elif not target.exists() or json.loads(target.read_text()) != manifest:
        raise SystemExit("Artifact manifest missing or stale")
    print(f"PASS: {len(files)} publication files checked, {len(manifest['artifacts'])} numerical artifacts validated")


if __name__ == "__main__":
    main()
