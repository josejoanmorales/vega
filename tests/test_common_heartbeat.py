"""Dead-man's-switch heartbeat: the only watcher that survives a wedge."""

import urllib.error
from typing import Any

import pytest

from vega.common import heartbeat


class _Resp:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *a: object) -> None:
        return None


@pytest.fixture(autouse=True)
def _reset_warning() -> None:
    heartbeat._warned = False


def _capture(monkeypatch, status: int = 200) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _fake(req, timeout=None):  # noqa: ANN001
        calls.append({"url": req.full_url, "data": req.data, "timeout": timeout})
        return _Resp(status)

    monkeypatch.setattr(heartbeat.urllib.request, "urlopen", _fake)
    return calls


def test_ok_start_and_fail_hit_the_right_endpoints(monkeypatch) -> None:
    monkeypatch.setenv(heartbeat.ENV_VAR, "https://hc.example/abc-123")
    calls = _capture(monkeypatch)

    assert heartbeat.ping() is True
    assert heartbeat.ping("start") is True
    assert heartbeat.ping("fail", "degraded") is True

    assert [c["url"] for c in calls] == [
        "https://hc.example/abc-123",
        "https://hc.example/abc-123/start",
        "https://hc.example/abc-123/fail",
    ]
    assert calls[2]["data"] == b"degraded"


def test_unconfigured_is_a_no_op_but_says_so_once(monkeypatch, capsys) -> None:
    """Silence is the failure mode under repair — an unwatched pipeline must
    announce that it is unwatched, but must not spam every run."""
    monkeypatch.delenv(heartbeat.ENV_VAR, raising=False)
    calls = _capture(monkeypatch)

    assert heartbeat.ping("start") is False
    assert heartbeat.ping() is False

    assert calls == []  # no network attempted
    out = capsys.readouterr().out
    assert out.count(heartbeat.ENV_VAR) == 1  # warned once, not per ping


def test_network_failure_never_raises_and_reports(monkeypatch, capsys) -> None:
    """A monitoring call must not be able to break the pipeline it watches."""
    monkeypatch.setenv(heartbeat.ENV_VAR, "https://hc.example/abc-123")

    def _boom(req, timeout=None):  # noqa: ANN001
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(heartbeat.urllib.request, "urlopen", _boom)

    assert heartbeat.ping("start") is False  # returned, did not raise
    assert "heartbeat" in capsys.readouterr().out


def test_http_error_status_is_a_failed_ping(monkeypatch) -> None:
    monkeypatch.setenv(heartbeat.ENV_VAR, "https://hc.example/abc-123")
    _capture(monkeypatch, status=500)
    assert heartbeat.ping() is False


def test_the_secret_url_is_never_logged_in_full(monkeypatch, capsys) -> None:
    """The ping URL is a bearer secret; run logs are read and pasted around."""
    ping_url = "https://hc.example/9f3c1d2e-private-token-value"
    monkeypatch.setenv(heartbeat.ENV_VAR, ping_url)

    def _boom(req, timeout=None):  # noqa: ANN001
        raise OSError("connection reset")

    monkeypatch.setattr(heartbeat.urllib.request, "urlopen", _boom)
    heartbeat.ping()

    out = capsys.readouterr().out
    assert ping_url not in out
    assert "9f3c1d2e-private-token-value" not in out


def test_non_http_schemes_are_refused(monkeypatch) -> None:
    """The URL comes from .env; urlopen would otherwise honour file:// and
    turn a typo'd or tampered config into a local-file read."""
    monkeypatch.setenv(heartbeat.ENV_VAR, "file:///etc/passwd")
    calls = _capture(monkeypatch)
    assert heartbeat.ping() is False
    assert calls == []
