"""Local web UI HTTP server (WI-088): stdlib only, bound to 127.0.0.1 ONLY —
no auth exists, so binding wider would expose real order-triggering machinery
to the local network. Matches Caral's own core-server pattern (a thin
BaseHTTPRequestHandler + one static page), scoped down for a single operator.

Trigger safety (WI-088 review):
- POST /api/run requires the `X-Vega-Run: 1` header (sent by the page's own
  fetch). A cross-origin page cannot attach a custom header to a "simple"
  request — the browser forces a CORS preflight this server never answers —
  so drive-by `fetch('http://127.0.0.1:7788/api/run', {method:'POST'})` from
  a malicious tab dies in the browser. Host validation closes DNS rebinding.
- EVERY /api/run attempt is audited to `data/web-runs/audit.log` (UTC time,
  client address, User-Agent, Origin, Referer, outcome) — an unattributed
  trigger was observed during the first live smoke and could never be
  attributed after the fact; that must not be possible again.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vega.common.paths import DATA_ROOT
from vega.web import dashboard
from vega.web.markdown import render_markdown
from vega.web.runner import RunAlreadyInProgress, Runner

DEFAULT_PORT = 7788
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"
BRIEFINGS_DIR = DATA_ROOT / "briefings"
AUDIT_LOG = DATA_ROOT / "web-runs" / "audit.log"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
BRIEFING_PATH_RE = re.compile(r"^/api/briefings/(\d{4}-\d{2}-\d{2})$")
INSPECT_PATH_RE = re.compile(r"^/api/inspect/([A-Za-z0-9.\-]{1,10})$")
ALLOWED_HOSTS_RE = re.compile(r"^(127\.0\.0\.1|localhost)(:\d+)?$")
ALLOWED_ORIGIN_RE = re.compile(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$")

runner = Runner()


def _list_briefing_dates() -> list[str]:
    if not BRIEFINGS_DIR.exists():
        return []
    return sorted(p.stem for p in BRIEFINGS_DIR.glob("*.md") if DATE_RE.match(p.stem))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # quiet; /api/run has its own audit log
        pass

    def _send(self, code: int, body: object, ctype: str = "application/json") -> None:
        payload = body if isinstance(body, (bytes, str)) else json.dumps(body)
        if isinstance(payload, str):
            payload = payload.encode()
        self._responded = True
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _audit_run_attempt(self, outcome: str) -> None:
        """One durable JSON line per /api/run attempt — who, when, from what,
        and what happened. The first live smoke produced a real run nobody
        could attribute; every future trigger must be."""
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_LOG.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "at": datetime.now(UTC).isoformat(),
                        "client": self.client_address[0],
                        "user_agent": self.headers.get("User-Agent"),
                        "origin": self.headers.get("Origin"),
                        "referer": self.headers.get("Referer"),
                        "outcome": outcome,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def _reject_cross_site(self) -> str | None:
        """The reason to refuse this POST, or None if it is legitimately ours."""
        host = self.headers.get("Host", "")
        if not ALLOWED_HOSTS_RE.match(host):
            return f"bad Host {host!r} (DNS rebinding?)"
        if self.headers.get("X-Vega-Run") != "1":
            return "missing X-Vega-Run header (cross-site or non-UI trigger)"
        origin = self.headers.get("Origin")
        if origin is not None and not ALLOWED_ORIGIN_RE.match(origin):
            return f"cross-site Origin {origin!r}"
        return None

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's naming contract
        path = self.path.split("?", 1)[0]
        try:
            if path == "/":
                return self._send(200, STATIC_INDEX.read_text(), "text/html; charset=utf-8")
            if path == "/api/status":
                return self._send(200, runner.status())
            if path == "/api/briefings":
                return self._send(200, _list_briefing_dates())
            if path == "/api/positions":
                return self._send(200, dashboard.positions())
            if path == "/api/signal-health":
                return self._send(200, dashboard.signal_health())
            if path == "/api/failures":
                return self._send(200, dashboard.failures())
            m = BRIEFING_PATH_RE.match(path)
            if m:
                briefing_date = m.group(1)  # regex-validated: date-shaped only, no traversal
                file_path = BRIEFINGS_DIR / f"{briefing_date}.md"
                if not file_path.is_file():
                    return self._send(404, {"error": "not found"})
                text = file_path.read_text()
                return self._send(200, {"date": briefing_date, "html": render_markdown(text)})
            m = INSPECT_PATH_RE.match(path)
            if m:
                symbol = m.group(1).upper()
                try:
                    return self._send(200, dashboard.inspect_symbol(symbol))
                except dashboard.SymbolNotInUniverse:
                    return self._send(404, {"error": f"{symbol!r} is not in the tradable universe"})
            self._send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001 — never crash the handler thread
            self._error(exc)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/run":
                rejection = self._reject_cross_site()
                if rejection is not None:
                    self._audit_run_attempt(f"rejected: {rejection}")
                    return self._send(403, {"error": rejection})
                try:
                    run_id = runner.start()
                except RunAlreadyInProgress as exc:
                    self._audit_run_attempt(f"refused-409: {exc}")
                    return self._send(409, {"error": str(exc)})
                self._audit_run_attempt(f"started: {run_id}")
                return self._send(202, {"run_id": run_id})
            self._send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._error(exc)

    def _error(self, exc: Exception) -> None:
        """Server-side fault: 500 — but only if we haven't already started a
        response (a second response into a half-written stream corrupts the
        protocol and re-raises on a broken socket)."""
        if not getattr(self, "_responded", False):
            self._send(500, {"error": str(exc)})


def serve(port: int = DEFAULT_PORT) -> None:
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        print(f"cannot bind port {port} ({exc}) — is another vega.web running? pass --port N")
        sys.exit(1)
    print(f"Vega web UI listening on http://127.0.0.1:{port}", flush=True)
    httpd.serve_forever()
