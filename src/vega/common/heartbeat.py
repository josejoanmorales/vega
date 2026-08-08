"""Dead-man's-switch heartbeat (WI-129 follow-up).

Every alerting mechanism Vega had before this one lived INSIDE the process
that fails, which is why the 2026-07-27 wedge ran silent for 3d13h: a hung
process sends nothing, and no later run started to notice (launchd will not
fire a `StartCalendarInterval` job while it still believes the previous
instance is running, so the stuck-run check in `vega.run` never got a chance
to run either).

The only observer that survives that is one that lives OUTSIDE the machine
and treats SILENCE as the alarm. This pings a dead-man's-switch service
(Healthchecks.io shape: `<url>` = ok, `<url>/start` = began, `<url>/fail` =
failed). Miss the expected window and the service alerts — which covers the
cases local alerting structurally cannot: a wedged process, launchd never
firing, the Mac asleep, the machine offline, power cut.

The `/start` ping is what actually catches a wedge: a run that starts and
never reports completion is exactly the incident's signature.

Configure `VEGA_HEALTHCHECK_URL` in the gitignored `.env`. The URL embeds a
secret token, so it is never logged in full.

PRIVACY: these messages leave the machine to a third party. Send operational
state only — never positions, prices, P&L, or account identifiers.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request

ENV_VAR = "VEGA_HEALTHCHECK_URL"
TIMEOUT_S = 10
_warned = False


def _redacted(url: str) -> str:
    """Enough to identify the endpoint, not enough to ping it."""
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return f"…/{tail[:4]}…" if tail else "…"


def ping(kind: str = "", detail: str = "") -> bool:
    """Report liveness. `kind` is "" (ok), "start", or "fail".

    Never raises and never blocks longer than TIMEOUT_S: a monitoring call
    must not be able to break the pipeline it monitors. Returns whether the
    ping was delivered, and says so on stderr when it was not — a heartbeat
    that fails silently is the same bug class it exists to fix.
    """
    global _warned
    base = os.environ.get(ENV_VAR, "").strip()
    if not base:
        if not _warned:
            _warned = True
            print(
                f"heartbeat: {ENV_VAR} unset — no external dead-man's switch is watching "
                "this pipeline (see common/heartbeat.py)",
                flush=True,
            )
        return False

    url = base.rstrip("/") + (f"/{kind}" if kind else "")
    # The URL comes from the environment, so pin the scheme: urlopen would
    # otherwise honour file:// (and friends), turning a typo'd or tampered
    # .env into a local-file read dressed up as monitoring.
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        print(f"heartbeat: refusing non-HTTP {ENV_VAR} scheme", flush=True)
        return False
    # Body is the service's log line for this ping. Operational state only.
    data = detail.encode()[:9000] if detail else None
    req = urllib.request.Request(url, data=data, method="POST")  # noqa: S310 — scheme pinned above
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # noqa: S310
            if resp.status >= 400:
                print(f"heartbeat: {_redacted(url)} returned {resp.status}", flush=True)
                return False
            return True
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"heartbeat: ping to {_redacted(url)} failed ({type(exc).__name__}: {exc})")
        return False
