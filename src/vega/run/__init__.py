"""The one locked pipeline entry point (WI-088): ingest + briefing under the
run lock, so the scheduled (launchd) and on-demand (web UI) triggers can never
run concurrently. `vega.run.main` is what both now call.
"""
