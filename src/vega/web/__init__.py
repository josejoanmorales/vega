"""Local web UI (WI-088): a Run button + status/log view + briefing reader,
bound strictly to 127.0.0.1 (no auth — solo machine, paper account; auth is a
Phase-2/SaaS concern, not built here). Zero third-party dependencies, matching
Caral's own core server pattern (`http.server` + one embedded HTML page).
"""
