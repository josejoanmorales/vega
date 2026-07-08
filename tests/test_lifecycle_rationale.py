from pathlib import Path

import pytest

from vega.lifecycle.rationale import RationaleRegistry


def test_no_rationale_before_recording(tmp_path: Path) -> None:
    reg = RationaleRegistry(tmp_path / "r.jsonl")
    assert reg.has_rationale("trend_pullback_v1") is False


def test_record_and_has_rationale(tmp_path: Path) -> None:
    reg = RationaleRegistry(tmp_path / "r.jsonl")
    reg.record(
        "trend_pullback_v1",
        "Momentum persists after a shallow pullback because...",
        author="human:jose",
    )
    assert reg.has_rationale("trend_pullback_v1") is True
    assert reg.has_rationale("other_family") is False


def test_empty_rationale_rejected(tmp_path: Path) -> None:
    reg = RationaleRegistry(tmp_path / "r.jsonl")
    with pytest.raises(ValueError, match="non-empty"):
        reg.record("fam", "   ", author="human:jose")


def test_correction_is_a_new_record_never_an_edit(tmp_path: Path) -> None:
    reg = RationaleRegistry(tmp_path / "r.jsonl")
    reg.record("fam", "v1 rationale", author="human:jose")
    reg.record("fam", "v2 corrected rationale", author="human:jose")
    history = reg.history("fam")
    assert len(history) == 2
    assert history[0]["text"] == "v1 rationale" and history[1]["text"] == "v2 corrected rationale"
