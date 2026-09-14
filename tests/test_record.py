"""scripts/record.py offline: market hours, the snapshot format, and loading it back."""
import importlib.util
import sys
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


def cboe_payload(chains, feed_time):
    """Cboe delayed-quotes JSON for {expiry: (calls, puts)} in the test_marketdata chain format."""
    options = []
    for expiry, (calls, puts) in chains.items():
        for df, cp in ((calls, "C"), (puts, "P")):
            for r in df.itertuples():
                root = r.contractSymbol.rstrip("0123456789CP")
                options.append({
                    "option": f"{root}{expiry[2:4]}{expiry[5:7]}{expiry[8:10]}{cp}{round(r.strike * 1000):08d}",
                    "bid": r.bid, "bid_size": 7.0, "ask": r.ask, "ask_size": 9.0, "iv": 0.2, "delta": 0.3,
                    "volume": r.volume, "open_interest": r.openInterest,
                    "last_trade_time": r.lastTradeDate.tz_convert("America/New_York").strftime("%Y-%m-%dT%H:%M:%S"),
                })
    return {"timestamp": feed_time.strftime("%Y-%m-%d %H:%M:%S"), "symbol": "_SPX",
            "data": {"current_price": SPOT, "options": options}}


def test_cboe_snapshot_round_trip_matches_a_slice_built_from_the_raw_chain(tmp_path):
    calls, puts = chain("SPXW")
    am_c, am_p = chain("SPX", spread=0.5)
    calls.loc[calls.index[:3], ["bid", "ask"]] = 0.0       # unquoted contracts are not stored
    payload = cboe_payload({EXPIRY: (pd.concat([calls, am_c]), pd.concat([puts, am_p])), "2026-12-18": (calls, puts)},
                           AS_OF.tz_convert("UTC") + pd.Timedelta(minutes=15))
    fetched = AS_OF + pd.Timedelta(minutes=16)
    board = record.tidy_cboe(payload, fetched)
    assert list(board.columns) == record.SNAPSHOT_COLUMNS
    n = len(STRIKES)
    assert len(board) == (4 * n - 3) + (2 * n - 3)
    assert set(board["root"]) == {"SPX", "SPXW"} and set(board["expiry"]) == {EXPIRY, "2026-12-18"}
    assert set(board["last_trade"]) == {int(AS_OF.timestamp())}
    assert (board["quote_time"] == AS_OF.isoformat()).all() and (board["fetched_at"] == fetched.isoformat()).all()
    assert (board["source"] == "cboe").all() and (board["bid_size"] == 7.0).all() and (board["cboe_delta"] == 0.3).all()
    path = tmp_path / "1616.csv.gz"
    board.to_csv(path, index=False, float_format="%.10g")

    got = load_snapshot(path, EXPIRY)
    want = build_slice(calls, puts, EXPIRY, SPOT, as_of=AS_OF)      # quote time: feed time minus the delay
    assert (got.expiry, got.as_of) == (want.expiry, want.as_of)
    assert got.forward == pytest.approx(want.forward, rel=1e-12)
    pd.testing.assert_frame_equal(got.quotes, want.quotes, rtol=1e-9)


def test_cboe_symbols_times_and_bad_input():
    payload = {"timestamp": "2026-09-14 17:15:31", "data": {"current_price": 2905.5, "options": [
        {"option": "RUTW260918C02912500", "bid": 1.0, "bid_size": 0.0, "ask": 1.2, "ask_size": 3.0, "iv": 0.2,
         "delta": 0.4, "volume": 0.0, "open_interest": 5.0, "last_trade_time": None},
        {"option": "RUT261218P00100000", "bid": 0.0, "bid_size": 0.0, "ask": 0.05, "ask_size": 1.0, "iv": 0.9,
         "delta": -0.01, "volume": 2.0, "open_interest": 5.0, "last_trade_time": "2026-09-14T13:00:10"},
        {"option": "RUT261218C00100000", "bid": 0.0, "bid_size": 0.0, "ask": 0.0, "ask_size": 0.0, "iv": 0.0,
         "delta": 0.0, "volume": 0.0, "open_interest": 0.0, "last_trade_time": None},   # no quote: dropped
    ]}}
    board = record.tidy_cboe(payload, pd.Timestamp("2026-09-14 17:16:01", tz="UTC"))
    assert board["root"].tolist() == ["RUTW", "RUT"] and board["expiry"].tolist() == ["2026-09-18", "2026-12-18"]
    assert board["strike"].tolist() == [2912.5, 100.0] and board["is_call"].tolist() == [True, False]
    assert np.isnan(board["last_trade"].iloc[0])
    assert board["last_trade"].iloc[1] == pd.Timestamp("2026-09-14 17:00:10", tz="UTC").timestamp()  # EDT = UTC-4
    assert board["quote_time"].iloc[0] == "2026-09-14T17:00:31+00:00" and board["spot"].iloc[0] == 2905.5
    payload["data"]["options"][0]["option"] = "RUTW2609C02912500"
    with pytest.raises(ValueError, match="unexpected Cboe symbols"):
        record.tidy_cboe(payload, pd.Timestamp("2026-09-14 17:16:01", tz="UTC"))
    with pytest.raises(ValueError, match="no options"):
        record.tidy_cboe({"timestamp": payload["timestamp"], "data": {"current_price": 1.0, "options": []}}, AS_OF)


def test_files_recorded_before_quote_time_load_with_the_fetch_time(tmp_path):
    calls, puts = chain("SPXW")
    board = record.tidy({EXPIRY: (calls, puts)}, SPOT, AS_OF).drop(columns=["quote_time", "source", "bid_size",
                                                                           "ask_size", "cboe_iv", "cboe_delta"])
    path = tmp_path / "old.csv.gz"
    board.to_csv(path, index=False, float_format="%.10g")
    assert load_snapshot(path, EXPIRY).as_of == AS_OF


def test_cboe_failure_falls_back_to_yahoo(tmp_path, monkeypatch, capsys):
    calls, puts = chain("SPXW")

    def broken(ticker):
        raise OSError("blocked")

    monkeypatch.setattr(record, "fetch_cboe", broken)
    monkeypatch.setattr(record, "fetch_board", lambda ticker: ({EXPIRY: (calls, puts)}, SPOT, AS_OF))
    monkeypatch.setattr(sys, "argv", ["record.py", "--force", "--out", str(tmp_path)])
    assert record.main() == 0
    assert "Cboe failed (OSError: blocked), using Yahoo" in capsys.readouterr().err
    got = pd.read_csv(tmp_path / "spx" / "2026-09-11" / "1600.csv.gz")
    assert (got["source"] == "yahoo").all() and got["bid_size"].isna().all() and len(got) == 2 * len(STRIKES)
    assert load_snapshot(tmp_path / "rut" / "2026-09-11" / "1600.csv.gz", EXPIRY).as_of == AS_OF


def test_traded_today_spots_holidays():
    now = datetime.fromisoformat("2026-09-14T15:00:00-04:00")
    today = pd.DataFrame({"last_trade": [pd.Timestamp("2026-09-14 14:59", tz="America/New_York").timestamp()]})
    friday = pd.DataFrame({"last_trade": [pd.Timestamp("2026-09-11 15:59", tz="America/New_York").timestamp()]})
    assert record.traded_today(today, now) and not record.traded_today(friday, now)
    assert not record.traded_today(pd.DataFrame({"last_trade": [np.nan]}), now)
