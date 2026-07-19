"""Local web UI HTTP server (WI-088): stdlib only, bound to 127.0.0.1 ONLY —
no auth exists, so binding wider would expose real order-triggering machinery
to the local network. Matches Caral's own core-server pattern (a thin
BaseHTTPRequestHandler + one static page), scoped down for a single operator.
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vega.common.paths import DATA_ROOT
from vega.web.markdown import render_markdown
from vega.web.runner import RunAlreadyInProgress, Runner

DEFAULT_PORT = 7788
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"
BRIEFINGS_DIR = DATA_ROOT / "briefings"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
BRIEFING_PATH_RE = re.compile(r"^/api/briefings/(\d{4}-\d{2}-\d{2})$")

runner = Runner()


def _list_briefing_dates() -> list[str]:
    if not BRIEFINGS_DIR.exists():
        return []
    return sorted(p.stem for p in BRIEFINGS_DIR.glob("*.md") if DATE_RE.match(p.stem))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # quiet
        pass

    def _send(self, code: int, body: object, ctype: str = "application/json") -> None:
        payload = body if isinstance(body, (bytes, str)) else json.dumps(body)
        if isinstance(payload, str):
            payload = payload.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's naming contract
        path = self.path.split("?", 1)[0]
        try:
            if path == "/":
                return self._send(200, STATIC_INDEX.read_text(), "text/html; charset=utf-8")
            if path == "/api/status":
                return self._send(200, runner.status())
            if path == "/api/briefings":
                return self._send(200, _list_briefing_dates())
            m = BRIEFING_PATH_RE.match(path)
            if m:
                date = m.group(1)  # regex-validated: date-shaped only, no traversal
                file_path = BRIEFINGS_DIR / f"{date}.md"
                if not file_path.is_file():
                    return self._send(404, {"error": "not found"})
                text = file_path.read_text()
                return self._send(200, {"date": date, "html": render_markdown(text)})
            self._send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001 — never crash the handler on a bad request
            self._send(400, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/run":
                try:
                    run_id = runner.start()
                except RunAlreadyInProgress as exc:
                    return self._send(409, {"error": str(exc)})
                return self._send(202, {"run_id": run_id})
            self._send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"error": str(exc)})


def serve(port: int = DEFAULT_PORT) -> None:
    print(f"Vega web UI listening on http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
