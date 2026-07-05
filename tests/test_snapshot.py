from pathlib import Path

import pandas as pd
import pytest

from vega.data.snapshot import write_clean
from vega.data.types import SnapshotConflictError

FRAME = pd.DataFrame({"symbol": ["AAPL"], "date": ["2026-07-02"], "close": [230.5]})


def test_write_once_then_identical_rewrite_is_noop(tmp_path: Path) -> None:
    p1 = write_clean("2026-07-02", "bars_equity", FRAME, root=tmp_path)
    p2 = write_clean("2026-07-02", "bars_equity", FRAME.copy(), root=tmp_path)
    assert p1 == p2 and p1.exists()


def test_rewrite_with_different_content_raises(tmp_path: Path) -> None:
    write_clean("2026-07-02", "bars_equity", FRAME, root=tmp_path)
    drifted = FRAME.assign(close=[231.0])
    with pytest.raises(SnapshotConflictError):
        write_clean("2026-07-02", "bars_equity", drifted, root=tmp_path)
