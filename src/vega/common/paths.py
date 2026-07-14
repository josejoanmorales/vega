"""Project-anchored paths — the ONE place filesystem roots are resolved.

A review found registry/rationale/store default paths were CWD-relative
(same defect class as the earlier universe_version bug): running any module
from outside the repo root silently created a SECOND divergent append-only
registry, which resets cumulative grid accounting and bypasses the rising
promotion bar. Anchoring to the package location (src layout, editable
install) makes every default path CWD-independent.

Solo-scale caveat (stated): parents[3] assumes the src/vega/common layout of
an editable install; a wheel install into site-packages would need an env
override — add VEGA_DATA_ROOT support then, not before.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"
