"""Index option chains from yfinance (SPX, RUT), turned into one clean smile per expiry.

A slice is a table of out-of-the-money quotes (puts below the forward, calls
at or above it) with mid prices and Black-76 implied vols, plus the forward
and the time to expiry. Everything except `fetch_chain` works on plain
DataFrames, so it is tested offline.

Choices:
- Forward from put-call parity, C - P = F - K (undiscounted, like black76.py):
  the median of C - P + K over the strikes closest to spot where both sides
  have a two-sided quote.
- Only the PM-settled weekly root (SPXW, RUTW: the root ending in W) when a
  date also has the AM-settled one (SPX, RUT), so each strike appears once and
  all quotes share one expiry time.
- Expiry at 16:00 New York time; quote time is the chain's latest trade.
  Year fraction in calendar days / 365.
- A quote is kept if bid > 0 and ask > bid. Implied vols come from the mid;
  quotes whose mid has no implied vol are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import black76

EXPIRY_HOUR = 16  # New York time
NEAR_SPOT = 10    # strikes either side of spot used for the forward

COLUMNS = ["strike", "is_call", "bid", "ask", "mid", "iv", "iv_bid", "iv_ask", "delta", "volume", "open_interest"]


@dataclass(frozen=True)
class Slice:
    """One expiry: out-of-the-money quotes sorted by strike, forward and tau."""

    expiry: str          # YYYY-MM-DD
    as_of: pd.Timestamp  # quote time, UTC
    forward: float
    tau: float           # years
    quotes: pd.DataFrame  # COLUMNS


def fetch_chain(ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw calls and puts from Yahoo. Needs the network."""
    import yfinance as yf

    chain = yf.Ticker(ticker).option_chain(expiry)
    return chain.calls, chain.puts


def year_fraction(expiry: str, as_of: pd.Timestamp) -> float:
    """Calendar years from as_of (UTC) to 16:00 New York time on the expiry date."""
    end = pd.Timestamp(f"{expiry} {EXPIRY_HOUR:02d}:00").tz_localize("America/New_York").tz_convert("UTC")
    return (end - as_of).total_seconds() / (365.0 * 86400.0)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """PM-settled contracts only (if a date has both), two-sided quotes only."""
    df = df[(df["bid"].fillna(0.0) > 0.0) & (df["ask"].fillna(0.0) > df["bid"].fillna(0.0))]
    # After the quote filter: a listed but unquoted PM root must not hide the AM quotes.
    root = df["root"] if "root" in df else (
        df["contractSymbol"].str.extract(r"^([A-Z]+)\d", expand=False) if "contractSymbol" in df else None)
    if root is not None:
        pm = root.str.endswith("W", na=False)
        if pm.any():
            df = df[pm]
    return df.drop_duplicates(subset="strike").sort_values("strike")


def forward_from_parity(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> float:
    """Median of C - P + K at the strikes nearest spot quoted on both sides."""
    both = _clean(calls).merge(_clean(puts), on="strike", suffixes=("_c", "_p"))
    if both.empty:
        raise ValueError("no strike has two-sided quotes for both a call and a put")
    both = both.iloc[np.argsort(np.abs(both["strike"].to_numpy() - spot))[: 2 * NEAR_SPOT]]
    mid_c = (both["bid_c"] + both["ask_c"]) / 2.0
    mid_p = (both["bid_p"] + both["ask_p"]) / 2.0
    return float(np.median(mid_c - mid_p + both["strike"]))


def build_slice(calls: pd.DataFrame, puts: pd.DataFrame, expiry: str, spot: float,
                as_of: pd.Timestamp | None = None) -> Slice:
    """Out-of-the-money quotes with implied vols. as_of defaults to the latest trade time."""
    if as_of is None:
        as_of = pd.concat([calls["lastTradeDate"], puts["lastTradeDate"]]).max()
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    tau = year_fraction(expiry, as_of)
    if not tau > 0.0:
        raise ValueError(f"expiry {expiry} is not after the quote time {as_of}")
    forward = forward_from_parity(calls, puts, spot)

    parts = []
    for df, is_call in ((_clean(calls), True), (_clean(puts), False)):
        otm = df[df["strike"] >= forward] if is_call else df[df["strike"] < forward]
        parts.append(pd.DataFrame({
            "strike": otm["strike"].astype(float),
            "is_call": is_call,
            "bid": otm["bid"].astype(float),
            "ask": otm["ask"].astype(float),
            "volume": otm.get("volume", pd.Series(np.nan, index=otm.index)).astype(float),
            "open_interest": otm.get("openInterest", pd.Series(np.nan, index=otm.index)).astype(float),
        }))
    q = pd.concat(parts).sort_values("strike").reset_index(drop=True)
    q["mid"] = (q["bid"] + q["ask"]) / 2.0
    for col, price in (("iv", "mid"), ("iv_bid", "bid"), ("iv_ask", "ask")):
        q[col] = black76.implied_vol(q[price].to_numpy(), forward, q["strike"].to_numpy(), tau, q["is_call"].to_numpy())
    q = q[np.isfinite(q["iv"])].reset_index(drop=True)
    q["delta"] = black76.delta(forward, q["strike"].to_numpy(), tau, q["iv"].to_numpy(), q["is_call"].to_numpy())
    return Slice(expiry, as_of, forward, tau, q[COLUMNS])


def nearest_expiry(expiries, as_of: pd.Timestamp, days: float) -> str:
    """The listed expiry closest to `days` calendar days after as_of."""
    return min(expiries, key=lambda e: abs(year_fraction(e, pd.Timestamp(as_of)) * 365.0 - days))


def save_slice(s: Slice, path) -> None:
    """One CSV: the quotes, with expiry, as_of, forward and tau repeated on each row."""
    out = s.quotes.copy()
    out.insert(0, "tau", s.tau)
    out.insert(0, "forward", s.forward)
    out.insert(0, "as_of", s.as_of.isoformat())
    out.insert(0, "expiry", s.expiry)
    out.to_csv(path, index=False, float_format="%.17g")


def load_snapshot(path, expiry: str) -> Slice:
    """One expiry of a whole-board snapshot written by scripts/record.py, built
    like any slice. The quote time is the fetch time (Yahoo's option quotes
    are typically delayed by about 15 minutes)."""
    df = pd.read_csv(path, dtype={"expiry": str, "root": str})
    d = df[df["expiry"] == expiry]
    if d.empty:
        raise LookupError(f"no {expiry} expiry in {path}")
    chain = d.rename(columns={"open_interest": "openInterest"})
    calls, puts = chain[chain["is_call"].astype(bool)], chain[~chain["is_call"].astype(bool)]
    return build_slice(calls, puts, expiry, float(df["spot"].iloc[0]), as_of=pd.Timestamp(df["fetched_at"].iloc[0]))


def load_slice(path) -> Slice:
    df = pd.read_csv(path)
    return Slice(str(df["expiry"].iloc[0]), pd.Timestamp(df["as_of"].iloc[0]), float(df["forward"].iloc[0]),
                 float(df["tau"].iloc[0]), df[COLUMNS].astype({c: bool if c == "is_call" else float for c in COLUMNS}))
