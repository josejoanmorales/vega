"""Vega daily pre-market briefing — v1 is a pure deterministic template.

Every figure comes from the engine (stored data); there is no LLM in this
loop, which satisfies the evidence-integrity guardrail by construction. An
optional cached LLM-rephrase layer may come later without changing the engine.
"""
