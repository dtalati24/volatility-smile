"""scripts/record.py offline: market hours, the snapshot format, and loading it back."""
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volsmile.marketdata import build_slice, load_snapshot
from test_marketdata import AS_OF, EXPIRY, SPOT, STRIKES, chain

spec = importlib.util.spec_from_file_location("record", Path(__file__).resolve().parent.parent / "scripts" / "record.py")
record = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record)


@pytest.mark.parametrize("when, open_", [
    ("2026-09-14 09:24", False),   # Monday, before the first run window
    ("2026-09-14 09:25", True),
    ("2026-09-14 12:00", True),
    ("2026-09-14 16:20", True),    # a late 16:00 run still records
    ("2026-09-14 16:21", False),
    ("2026-09-13 12:00", False),   # Sunday
    ("2026-12-14 12:00", True),    # winter time: same New York hours
])
def test_market_hours_are_new_york_time(when, open_):
    assert record.market_hours(pd.Timestamp(when, tz="America/New_York").to_pydatetime()) is open_
    assert record.market_hours(pd.Timestamp(when, tz="America/New_York").tz_convert("UTC").to_pydatetime()) is open_


def test_snapshot_round_trip_matches_a_slice_built_from_the_raw_chain(tmp_path):
    calls, puts = chain("SPXW")
    am_c, am_p = chain("SPX", spread=0.5)
    calls.loc[calls.index[:3], ["bid", "ask"]] = 0.0       # unquoted contracts are not stored
    board = record.tidy({EXPIRY: (pd.concat([calls, am_c]), pd.concat([puts, am_p])), "2026-12-18": (calls, puts)},
                        SPOT, AS_OF)
    assert list(board.columns) == record.SNAPSHOT_COLUMNS
    n = len(STRIKES)
    assert len(board) == (4 * n - 3) + (2 * n - 3)
    assert set(board["last_trade"]) == {int(AS_OF.timestamp())}
    path = tmp_path / "1600.csv.gz"
    board.to_csv(path, index=False, float_format="%.10g")

    got = load_snapshot(path, EXPIRY)
    want = build_slice(calls, puts, EXPIRY, SPOT, as_of=AS_OF)      # PM contracts win, as in the raw chain
    assert (got.expiry, got.as_of) == (want.expiry, want.as_of)
    assert got.forward == pytest.approx(want.forward, rel=1e-12)
    pd.testing.assert_frame_equal(got.quotes, want.quotes, rtol=1e-9)
    with pytest.raises(LookupError):
        load_snapshot(path, "2027-01-15")


def test_traded_today_spots_holidays():
    now = datetime.fromisoformat("2026-09-14T15:00:00-04:00")
    today = pd.DataFrame({"last_trade": [pd.Timestamp("2026-09-14 14:59", tz="America/New_York").timestamp()]})
    friday = pd.DataFrame({"last_trade": [pd.Timestamp("2026-09-11 15:59", tz="America/New_York").timestamp()]})
    assert record.traded_today(today, now) and not record.traded_today(friday, now)
    assert not record.traded_today(pd.DataFrame({"last_trade": [np.nan]}), now)
