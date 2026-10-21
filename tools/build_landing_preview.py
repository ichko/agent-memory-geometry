"""Export measured Atlas coordinates and compact memory inspection examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from html import escape
from pathlib import Path

if __package__:
    from .package_demo import redact_display
else:
    from package_demo import redact_display


def sample(items: list[int], limit: int) -> list[int]:
    if limit < 1:
        raise ValueError("Sample limits must be positive")
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    return [items[i * (len(items) - 1) // (limit - 1)] for i in range(limit)]


def render_preview(atlas: dict, turn_limit: int = 1500, memory_limit: int = 100, path_limit: int = 24) -> str:
    points = atlas["points"]
    sessions = atlas["sessions"]
    turn_ids = sorted({turn["p"] for session in sessions for turn in session["turns"]})
    memory_ids = sorted({p for history in atlas["histories"] for p in history["memories"].get("m2", [])})
    if not turn_ids:
        raise ValueError("The Atlas has no source turns")
    shown_turns = sample(turn_ids, turn_limit)
    shown_memories = sample(memory_ids, memory_limit)
    path_ids = sample([i for i, session in enumerate(sessions) if len(session["turns"]) > 1], path_limit)
    bounds_ids = sorted(set(turn_ids + memory_ids))
    xy = {i: points[i]["umap"] for i in bounds_ids}
    if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in xy.values()):
        raise ValueError("The preview needs finite measured 2D UMAP coordinates")
    xmin, xmax = min(p[0] for p in xy.values()), max(p[0] for p in xy.values())
    ymin, ymax = min(p[1] for p in xy.values()), max(p[1] for p in xy.values())
    unit = max((xmax - xmin) / 270, (ymax - ymin) / 220, 1e-6)
    pad = unit * 10
    source = atlas.get("correspondence", {}).get("fingerprint")
    provenance = {"dataset": atlas["title"], "projection": "umap", "settings": atlas["projections"]["umap"],
                  "embedding": atlas["embedding"], "atlasFingerprint": source, "turnSamples": len(shown_turns),
                  "memorySamples": len(shown_memories), "memorySystem": "m2", "sessionPaths": len(path_ids),
                  "sampling": "Evenly spaced point/session IDs; no label-based selection",
                  "coordinateHash": hashlib.sha256(json.dumps([[i, xy[i]] for i in bounds_ids], separators=(",", ":")).encode()).hexdigest()}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{xmin-pad:.6f} {-ymax-pad:.6f} {xmax-xmin+2*pad:.6f} {ymax-ymin+2*pad:.6f}" role="img" aria-labelledby="preview-title preview-desc">',
             '<title id="preview-title">Real LongMemEval conversations and sampled memories</title>',
             f'<desc id="preview-desc">{len(shown_turns)} source-turn samples, {len(shown_memories)} Ext-Dense memory samples, and {len(path_ids)} conversation paths from the existing shared UMAP fit. Each path stays within one session. This is a projection preview, not a memory-quality score.</desc>',
             f'<metadata>{escape(json.dumps(provenance, separators=(",", ":")))}</metadata>',
             '<g fill="none" stroke="#9a9da2" stroke-opacity=".22" stroke-width=".35" vector-effect="non-scaling-stroke">']
    for i in path_ids:
        coordinates = " ".join(f'{xy[t["p"]][0]:.6f},{-xy[t["p"]][1]:.6f}' for t in sessions[i]["turns"])
        lines.append(f'<polyline data-session-id="{escape(sessions[i]["id"])}" points="{coordinates}" vector-effect="non-scaling-stroke"/>')
    lines.extend(['</g>', '<g fill="#1f77b4">'])
    for i in shown_turns:
        x, y = xy[i]
        lines.append(f'<circle data-point-id="{i}" cx="{x:.6f}" cy="{-y:.6f}" r="{unit*.8:.6f}" fill-opacity=".48"/>')
    lines.extend(['</g>', '<g fill="#ff7f0e">'])
    for i in shown_memories:
        x, y = xy[i]
        lines.append(f'<circle data-point-id="{i}" cx="{x:.6f}" cy="{-y:.6f}" r="{unit*1.25:.6f}" fill-opacity=".8"/>')
    lines.extend(['</g>', '</svg>'])
    return "\n".join(lines) + "\n"


def memory_matches(atlas: dict, memories: list[int], folder: Path) -> dict[int, dict]:
    """Rank source turns within each memory's first containing history."""
    owners = {}
    for h, history in enumerate(atlas["histories"]):
        for p in history["memories"].get("m2", []):
            owners.setdefault(p, h)
    groups: dict[int, list[int]] = {}
    for p in memories:
        groups.setdefault(owners[p], []).append(p)
    result = {}
    for h, selected in groups.items():
        history = atlas["histories"][h]
        matrix = json.loads((folder / f"h{h + 1}.json").read_text())
        rows = [[s, t, turn["p"]] for s in history["sessions"] for t, turn in enumerate(atlas["sessions"][s]["turns"])]
        block = matrix.get("systems", {}).get("m2", {})
        columns = history["memories"]["m2"]
        if (matrix.get("schemaVersion") != 1 or matrix.get("history") != h
                or matrix.get("fingerprint") != atlas.get("correspondence", {}).get("fingerprint")
                or matrix.get("metric") != "cosine" or matrix.get("dimensions") != atlas["embedding"]["dimensions"]
                or matrix.get("rows") != rows or block.get("memories") != columns
                or len(block.get("scores", [])) != len(rows) * len(columns)
                or any(not math.isfinite(v) or not -1 <= v <= 1 for v in block.get("scores", []))):
            raise ValueError(f"Memory similarities do not match the atlas: h{h + 1}")
        for p in selected:
            column = columns.index(p)
            scores = [block["scores"][r * len(columns) + column] for r in range(len(rows))]
            ranked = sorted(range(len(rows)), key=lambda r: (-scores[r], r))[:3]
            result[p] = {"text": redact_display(atlas["points"][p]["text"]), "history": h,
                         "matches": [{"session": rows[r][0], "point": rows[r][2], "score": scores[r]} for r in ranked]}
    return result


def build_preview_data(atlas: dict, turn_limit: int = 6000, memory_limit: int = 400, path_limit: int = 36,
                       correspondence_dir: Path | None = None) -> dict:
    points, sessions = atlas["points"], atlas["sessions"]
    turns = sample(sorted({t["p"] for s in sessions for t in s["turns"]}), turn_limit)
    memories = sample(sorted({p for h in atlas["histories"] for p in h["memories"].get("m2", [])}), memory_limit)
    background_sessions = sample([i for i, s in enumerate(sessions) if len(s["turns"]) > 1], path_limit)
    details = memory_matches(atlas, memories, correspondence_dir) if correspondence_dir is not None else {}
    path_sessions = list(dict.fromkeys(background_sessions + [m["session"] for d in details.values() for m in d["matches"]]))
    paths = [sessions[i] for i in path_sessions]
    ids = sorted(set(turns + memories + [t["p"] for s in paths for t in s["turns"]]))
    if not ids:
        raise ValueError("The preview needs measured points")
    index = {p: i for i, p in enumerate(ids)}
    path_index = {s: i for i, s in enumerate(path_sessions)}
    layouts = {}
    candidates = [("umap3d", "3D UMAP", 3, atlas.get("projections3d", {}).get("umap"))]
    for key, label, dimension, settings in candidates:
        if settings is None:
            continue
        coordinates = [points[i].get(key) for i in ids]
        if any(p is None or len(p) != dimension or not all(math.isfinite(v) for v in p) for p in coordinates):
            raise ValueError(f"Missing or invalid measured coordinates: {key}")
        layouts[key] = {"label": label, "dimensions": dimension, "coordinates": coordinates, "settings": settings}
    if not layouts:
        raise ValueError("The preview has no measured layouts")
    return {"schemaVersion": 1, "dataset": atlas["title"], "embedding": atlas["embedding"],
            "atlasFingerprint": atlas.get("correspondence", {}).get("fingerprint"),
            "sampling": "Evenly spaced point/session IDs, plus paths for within-history cosine matches; no correctness labels",
            "pointIds": ids, "turns": [index[i] for i in turns], "memories": [index[i] for i in memories],
            "paths": [[index[t["p"]] for t in s["turns"]] for s in paths],
            "backgroundPaths": [path_index[s] for s in background_sessions],
            "memoryDetails": [{"point": index[p], "text": detail["text"], "history": detail["history"],
                               "matches": [{"path": path_index[m["session"]], "point": index[m["point"]], "score": m["score"]}
                                           for m in detail["matches"]]} for p, detail in details.items()],
            "matchMethod": "Top 3 source turns by original-space cosine in the memory's first containing history; candidate matches, not verified provenance",
            "layouts": layouts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, default=Path("outputs/atlas/atlas.json"))
    parser.add_argument("--output", type=Path, default=Path("site/atlas-preview.svg"))
    parser.add_argument("--data-output", type=Path, default=Path("site/preview-data.mjs"))
    args = parser.parse_args()
    atlas = json.loads(args.atlas.read_text())
    args.output.write_text(render_preview(atlas))
    args.data_output.write_text("// Measured Atlas coordinates, redacted memory display text, and original-space cosine matches. No source-turn text or vectors.\nexport const PREVIEW = " + json.dumps(build_preview_data(atlas, correspondence_dir=args.atlas.parent / "correspondence"), separators=(",", ":"), allow_nan=False) + ";\n")
    print(f"Wrote measured UMAP preview: {args.output.stat().st_size:,} bytes; no source-turn text or embedding calls")
    print(f"Interactive sample: {args.data_output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
