import numpy as np
import pandas as pd
import pytest

from volsmile import black76
from volsmile.marketdata import build_slice, forward_from_parity, load_slice, nearest_expiry, save_slice, year_fraction

F_TRUE, SPOT = 5012.5, 5000.0
AS_OF = pd.Timestamp("2026-09-11 20:00", tz="UTC")          # 16:00 New York, Friday
EXPIRY = "2026-10-16"
TAU = 35 / 365.0                                          # 16:00 to 16:00 New York
STRIKES = np.arange(4000.0, 6001.0, 25.0)


def smile(K):
    x = np.log(K / F_TRUE)
    return 0.16 - 0.25 * x + 0.9 * x**2


def chain(symbol="SPXW", spread=0.02):
    """Synthetic chain priced exactly off F_TRUE and smile(), with a small spread."""
    rows = {}
    for is_call in (True, False):
        mid = black76.price(F_TRUE, STRIKES, TAU, smile(STRIKES), is_call)
        half = np.maximum(spread * mid, 0.05) / 2
        rows[is_call] = pd.DataFrame({
            "contractSymbol": [f"{symbol}{int(k)}{'C' if is_call else 'P'}" for k in STRIKES],
            "strike": STRIKES, "bid": mid - half, "ask": mid + half,
            "volume": 10.0, "openInterest": 100.0, "lastTradeDate": AS_OF,
        })
    return rows[True], rows[False]


def test_year_fraction_is_new_york_1600_in_calendar_days():
    assert year_fraction(EXPIRY, AS_OF) == pytest.approx(TAU, rel=1e-12)
    # Across the November clock change New York moves from UTC-4 to UTC-5.
    assert year_fraction("2026-11-13", AS_OF) == pytest.approx((63 + 1 / 24) / 365.0, rel=1e-12)


def test_forward_from_parity_recovers_the_forward():
    calls, puts = chain()
    assert forward_from_parity(calls, puts, SPOT) == pytest.approx(F_TRUE, abs=1e-9)


def test_slice_recovers_the_smile_with_out_of_the_money_quotes_only():
    calls, puts = chain()
    s = build_slice(calls, puts, EXPIRY, SPOT)
    assert s.forward == pytest.approx(F_TRUE, abs=1e-9)
    assert s.tau == pytest.approx(TAU, rel=1e-12)
    q = s.quotes
    assert np.all(np.diff(q["strike"]) > 0)
    assert np.all(q.loc[q["is_call"], "strike"] >= s.forward)
    assert np.all(q.loc[~q["is_call"], "strike"] < s.forward)
    liquid = q["mid"] > 1.0   # tiny prices lose their vol to the price rounding in the spread floor
    np.testing.assert_allclose(q.loc[liquid, "iv"], smile(q.loc[liquid, "strike"]), rtol=1e-3)
    assert np.all(q["iv_bid"].fillna(0) <= q["iv"]) and np.all(q["iv"] <= q["iv_ask"])
    np.testing.assert_allclose(
        q["delta"], black76.delta(s.forward, q["strike"].to_numpy(), s.tau, q["iv"].to_numpy(), q["is_call"].to_numpy()))


def test_bad_quotes_are_dropped():
    calls, puts = chain()
    puts.loc[puts["strike"] == 4500.0, "bid"] = 0.0          # no bid
    puts.loc[puts["strike"] == 4600.0, "ask"] = puts.loc[puts["strike"] == 4600.0, "bid"]  # locked
    calls.loc[calls["strike"] == 5500.0, "bid"] = np.nan
    calls.loc[calls["strike"] == 5600.0, ["bid", "ask"]] = [1e6, 1e6 + 1]  # above the forward: no vol
    s = build_slice(calls, puts, EXPIRY, SPOT)
    assert not s.quotes["strike"].isin([4500.0, 4600.0, 5500.0, 5600.0]).any()


@pytest.mark.parametrize("pm, am", [("SPXW", "SPX"), ("RUTW", "RUT")])
def test_pm_settled_contracts_win_when_both_are_listed(pm, am):
    c_pm, p_pm = chain(pm)
    c_am, p_am = chain(am, spread=0.5)
    c_am["bid"] *= 0.5                                     # AM quotes would give different vols
    p_am["bid"] *= 0.5
    s = build_slice(pd.concat([c_am, c_pm]), pd.concat([p_am, p_pm]), EXPIRY, SPOT)
    assert s.quotes["strike"].is_unique
    liquid = s.quotes["mid"] > 1.0
    np.testing.assert_allclose(s.quotes.loc[liquid, "iv"], smile(s.quotes.loc[liquid, "strike"]), rtol=1e-3)


def test_unquoted_pm_contracts_do_not_hide_am_quotes():
    c_pm, p_pm = chain("RUTW")
    c_am, p_am = chain("RUT")
    s = build_slice(pd.concat([c_am, c_pm.assign(bid=0.0)]), pd.concat([p_am, p_pm.assign(bid=0.0)]), EXPIRY, SPOT)
    assert len(s.quotes) > 0.9 * len(STRIKES)


def test_expired_or_empty_chains_raise():
    calls, puts = chain()
    with pytest.raises(ValueError):
        build_slice(calls, puts, "2026-09-11", SPOT, as_of=pd.Timestamp("2026-09-11 21:00", tz="UTC"))
    with pytest.raises(ValueError):
        build_slice(calls.assign(bid=0.0), puts, EXPIRY, SPOT)


def test_nearest_expiry():
    assert nearest_expiry(["2026-09-18", "2026-10-16", "2026-12-18"], AS_OF, 30) == "2026-10-16"


def test_save_and_load_round_trip(tmp_path):
    calls, puts = chain()
    s = build_slice(calls, puts, EXPIRY, SPOT)
    save_slice(s, tmp_path / "slice.csv")
    back = load_slice(tmp_path / "slice.csv")
    assert (back.expiry, back.as_of, back.forward, back.tau) == (s.expiry, s.as_of, s.forward, s.tau)
    pd.testing.assert_frame_equal(back.quotes, s.quotes)
