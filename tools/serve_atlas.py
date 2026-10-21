"""Serve the paper page, viewer, and ignored atlas data on localhost only."""

from __future__ import annotations

import argparse
import json
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("outputs/atlas/atlas.json"))
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    site = Path(__file__).resolve().parents[1] / "site"
    data = args.data.resolve()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **kwargs):
            super().__init__(*handler_args, directory=str(site), **kwargs)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def send_head(self):
            path = urlsplit(self.path).path
            if re.fullmatch(r"/correspondence/h[1-9][0-9]*\.json", path):
                matrix = data.parent / "correspondence" / path.rsplit("/", 1)[1]
                if not matrix.is_file():
                    self.send_error(404, "Build correspondence matrices with --correspondence")
                    return None
                stream = matrix.open("rb")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(matrix.stat().st_size))
                self.end_headers()
                return stream
            if path == "/atlas.json":
                if not data.is_file():
                    self.send_error(404, "The local atlas is still being prepared")
                    return None
                stream = data.open("rb")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(data.stat().st_size))
                self.end_headers()
                return stream
            if path == "/build-status.json":
                import io
                status = data.parent / "build-status.json"
                payload = status.read_bytes() if status.is_file() else json.dumps({"message": "Preparing the local real-data atlas. Both UMAP and PCA are computed before loading."}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                return io.BytesIO(payload)
            return super().send_head()

    print(f"Paper page: http://127.0.0.1:{args.port}/", flush=True)
    print(f"Memory Atlas: http://127.0.0.1:{args.port}/atlas.html", flush=True)
    print("Local data is served separately from the publishable site directory.", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
