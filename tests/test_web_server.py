import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from vega.web import server as server_module
from vega.web.runner import Runner
from vega.web.server import Handler


@pytest.fixture
def live_server(tmp_path: Path, monkeypatch):
    briefings_dir = tmp_path / "briefings"
    briefings_dir.mkdir()
    (briefings_dir / "2026-07-15.md").write_text("# Old\n")
    (briefings_dir / "2026-07-16.md").write_text("# New\n\n**bold**\n")
    monkeypatch.setattr(server_module, "BRIEFINGS_DIR", briefings_dir)
    monkeypatch.setattr(server_module, "runner", Runner(runs_dir=tmp_path / "runs"))

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        httpd.shutdown()
        thread.join(timeout=2)


def _get(port: int, path: str) -> tuple[int, dict | str]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read().decode()
    conn.close()
    ctype = resp.getheader("Content-Type", "")
    return resp.status, json.loads(body) if "json" in ctype else body


def _post(port: int, path: str, headers: dict | None = None) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    # the page's own fetch sends X-Vega-Run; default it so happy-path tests pass
    conn.request("POST", path, headers=headers if headers is not None else {"X-Vega-Run": "1"})
    resp = conn.getresponse()
    body = json.loads(resp.read().decode())
    conn.close()
    return resp.status, body


def test_index_page_served(live_server: int) -> None:
    status, body = _get(live_server, "/")
    assert status == 200
    assert "<html" in body.lower()


def test_status_idle_before_any_run(live_server: int) -> None:
    status, body = _get(live_server, "/api/status")
    assert status == 200
    assert body == {"state": "idle"}


def test_briefings_list_sorted(live_server: int) -> None:
    status, body = _get(live_server, "/api/briefings")
    assert status == 200
    assert body == ["2026-07-15", "2026-07-16"]


def test_briefing_content_rendered_to_html(live_server: int) -> None:
    status, body = _get(live_server, "/api/briefings/2026-07-16")
    assert status == 200
    assert body["date"] == "2026-07-16"
    assert "<h1>New</h1>" in body["html"] and "<b>bold</b>" in body["html"]


def test_unknown_briefing_date_404s(live_server: int) -> None:
    status, body = _get(live_server, "/api/briefings/2099-01-01")
    assert status == 404


def test_path_traversal_is_rejected(live_server: int) -> None:
    # the date-shaped regex route match structurally excludes traversal —
    # anything not matching /api/briefings/YYYY-MM-DD 404s as unknown
    status, _ = _get(live_server, "/api/briefings/../../../etc/passwd")
    assert status == 404


def test_run_returns_202_and_run_id(live_server: int, monkeypatch) -> None:
    # never spawn the REAL pipeline (real orders, real network) from a unit test
    monkeypatch.setattr(server_module.runner, "start", lambda: "fake-run-id")
    status, body = _post(live_server, "/api/run")
    assert status == 202
    assert body == {"run_id": "fake-run-id"}


def test_second_run_while_running_returns_409(live_server: int, monkeypatch) -> None:
    from vega.web.runner import RunAlreadyInProgress

    def _raise() -> str:
        raise RunAlreadyInProgress("busy")

    monkeypatch.setattr(server_module.runner, "start", _raise)
    status, body = _post(live_server, "/api/run")
    assert status == 409
    assert "busy" in body["error"]


def test_unknown_get_path_404s(live_server: int) -> None:
    status, _ = _get(live_server, "/api/nope")
    assert status == 404


# ---- WI-088 review-fix regressions ------------------------------------------


def test_run_without_custom_header_is_rejected_403(live_server: int) -> None:
    # CSRF drive-by: a POST without X-Vega-Run (what a cross-site simple
    # request looks like) must never trigger a real pipeline.
    status, body = _post(live_server, "/api/run", headers={})
    assert status == 403
    assert "X-Vega-Run" in body["error"]


def test_run_with_cross_site_origin_is_rejected_403(live_server: int) -> None:
    status, body = _post(
        live_server, "/api/run", headers={"X-Vega-Run": "1", "Origin": "http://evil.example"}
    )
    assert status == 403
    assert "Origin" in body["error"]


def test_run_attempts_are_audited(live_server: int, tmp_path: Path, monkeypatch) -> None:
    audit = tmp_path / "audit.log"
    monkeypatch.setattr(server_module, "AUDIT_LOG", audit)
    monkeypatch.setattr(server_module.runner, "start", lambda: "audited-run")
    _post(live_server, "/api/run")
    _post(live_server, "/api/run", headers={})  # rejected
    lines = [json.loads(x) for x in audit.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["outcome"] == "started: audited-run"
    assert "rejected" in lines[1]["outcome"]
    assert all("at" in ln and "client" in ln for ln in lines)
