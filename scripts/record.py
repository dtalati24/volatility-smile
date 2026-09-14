"""Record whole SPX and RUT option boards from Cboe, or Yahoo if Cboe fails (needs the network).

    python scripts/record.py                      # during US market hours only
    python scripts/record.py --force              # record now, whatever the time
    python scripts/record.py --out live/snapshots

Writes <out>/<spx|rut>/<YYYY-MM-DD>/<HHMM>.csv.gz, named by New York time, one
row per contract with a quote (ask > 0) on every listed expiry. Columns are
SNAPSHOT_COLUMNS; `volsmile.marketdata.load_snapshot` turns one expiry of a file
into a slice. Run every 15 minutes by .github/workflows/record.yml, which saves
the files on the data-live branch.

Cboe's delayed-quotes feed gives the whole board in one request, with the size
at the best bid and ask, Cboe's implied vol and delta. It is about 15 minutes
behind, so the quote time is the feed's timestamp minus CBOE_DELAY. Yahoo rows
have no sizes, vol or delta, and use the fetch time as the quote time.

Outside 09:25-16:20 New York time on weekdays it records nothing (the workflow
runs on a UTC schedule wide enough for both summer and winter time, and GitHub
often starts scheduled runs late). On a market holiday it records nothing
either: no SPX option traded today.

Only pandas and yfinance are needed, not the volsmile package, so the workflow
installs little.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, time as clock
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

TICKERS = {"^SPX": "spx", "^RUT": "rut"}
NY = ZoneInfo("America/New_York")
OPEN, CLOSE = clock(9, 25), clock(16, 20)
CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_{}.json"
CBOE_DELAY = pd.Timedelta(minutes=15)  # measured 2026-09-14: latest trade 15 min 2-5 s before the feed timestamp
SNAPSHOT_COLUMNS = ["fetched_at", "quote_time", "source", "spot", "expiry", "root", "strike", "is_call",
                    "bid", "ask", "bid_size", "ask_size", "volume", "open_interest", "last_trade",
                    "cboe_iv", "cboe_delta"]
RETRIES = 3


def market_hours(now: datetime) -> bool:
    """Weekday, between OPEN and CLOSE New York time."""
    ny = now.astimezone(NY)
    return ny.weekday() < 5 and OPEN <= ny.time() <= CLOSE


def _unix(ts: pd.Series) -> pd.Series:
    # Not astype(int64): its unit depends on the pandas datetime resolution.
    return (ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)


def _board(rows: pd.DataFrame, fetched_at: pd.Timestamp, quote_time: pd.Timestamp, source: str,
           spot: float) -> pd.DataFrame:
    """rows in SNAPSHOT_COLUMNS order; columns a source lacks are left empty."""
    out = rows.reindex(columns=SNAPSHOT_COLUMNS)
    out["fetched_at"], out["quote_time"] = fetched_at.isoformat(), quote_time.isoformat()
    out["source"], out["spot"] = source, spot
    return out


def tidy(chains: dict[str, tuple[pd.DataFrame, pd.DataFrame]], spot: float, fetched_at: pd.Timestamp) -> pd.DataFrame:
    """Yahoo chains: one row per quoted contract, in SNAPSHOT_COLUMNS. last_trade is Unix seconds (compact)."""
    parts = []
    for expiry, (calls, puts) in chains.items():
        for df, is_call in ((calls, True), (puts, False)):
            df = df[df["ask"].fillna(0.0) > 0.0]
            parts.append(pd.DataFrame({
                "expiry": expiry,
                "root": df["contractSymbol"].str.extract(r"^([A-Z]+)\d", expand=False),
                "strike": df["strike"].astype(float),
                "is_call": is_call,
                "bid": df["bid"].fillna(0.0).astype(float),
                "ask": df["ask"].astype(float),
                "volume": df["volume"].astype(float),
                "open_interest": df["openInterest"].astype(float),
                "last_trade": _unix(pd.to_datetime(df["lastTradeDate"], utc=True)),
            }))
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return _board(rows, fetched_at, fetched_at, "yahoo", spot)


def tidy_cboe(payload: dict, fetched_at: pd.Timestamp) -> pd.DataFrame:
    """Cboe delayed-quotes JSON: one row per quoted contract, in SNAPSHOT_COLUMNS.
    Symbols look like SPXW260928P07260000: root, YYMMDD, C/P, strike x 1000.
    last_trade_time is New York time; the feed timestamp is UTC."""
    o = pd.DataFrame(payload["data"]["options"])
    if o.empty:
        raise ValueError("Cboe returned no options")
    o = o[o["ask"].fillna(0.0) > 0.0]
    sym = o["option"].str.extract(r"^([A-Z]+)(\d\d)(\d\d)(\d\d)([CP])(\d{8})$")
    if sym.isna().any(axis=None):
        raise ValueError(f"unexpected Cboe symbols, e.g. {o['option'][sym[0].isna()].iloc[0]}")
    traded = pd.to_datetime(o["last_trade_time"]).dt.tz_localize(NY, ambiguous="NaT", nonexistent="NaT")
    rows = pd.DataFrame({
        "expiry": "20" + sym[1] + "-" + sym[2] + "-" + sym[3],
        "root": sym[0],
        "strike": sym[5].astype(int) / 1000.0,
        "is_call": sym[4] == "C",
        "bid": o["bid"].fillna(0.0).astype(float),
        "ask": o["ask"].astype(float),
        "bid_size": o["bid_size"].astype(float),
        "ask_size": o["ask_size"].astype(float),
        "volume": o["volume"].astype(float),
        "open_interest": o["open_interest"].astype(float),
        "last_trade": _unix(traded.dt.tz_convert("UTC")),
        "cboe_iv": o["iv"].astype(float),
        "cboe_delta": o["delta"].astype(float),
    })
    quote_time = pd.Timestamp(payload["timestamp"], tz="UTC") - CBOE_DELAY
    return _board(rows, fetched_at, quote_time, "cboe", float(payload["data"]["current_price"]))


def fetch_cboe(ticker: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    """The whole board in one request, tried RETRIES times."""
    req = urllib.request.Request(CBOE_URL.format(ticker.lstrip("^")), headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(RETRIES):
        fetched_at = pd.Timestamp.now(tz="UTC").floor("s")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return tidy_cboe(json.load(r), fetched_at), fetched_at
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(5)


def fetch_board(ticker: str) -> tuple[dict, float, pd.Timestamp]:
    """Every expiry's chain from Yahoo. An expiry that still fails after RETRIES tries is skipped."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    fetched_at = pd.Timestamp.now(tz="UTC").floor("s")
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    chains, failed = {}, []
    expiries = tk.options
    for expiry in expiries:
        for attempt in range(RETRIES):
            try:
                chain = tk.option_chain(expiry)
                chains[expiry] = (chain.calls, chain.puts)
                break
            except Exception as e:  # network or Yahoo hiccup: try again, then skip this expiry
                if attempt == RETRIES - 1:
                    failed.append(f"{expiry} ({type(e).__name__})")
                time.sleep(2)
    if failed:
        print(f"{ticker}: skipped {len(failed)} of {len(expiries)} expiries: {', '.join(failed)}", file=sys.stderr)
    if len(failed) > len(expiries) / 2:
        raise RuntimeError(f"{ticker}: most expiries failed")
    return chains, spot, fetched_at


def traded_today(board: pd.DataFrame, now: datetime) -> bool:
    last = pd.to_datetime(board["last_trade"], unit="s", utc=True).max()
    return pd.notna(last) and last.tz_convert(NY).date() == now.astimezone(NY).date()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "snapshots"))
    parser.add_argument("--force", action="store_true", help="record outside market hours and on holidays")
    opts = parser.parse_args()
    now = datetime.now(tz=NY)
    if not opts.force and not market_hours(now):
        print(f"{now:%a %Y-%m-%d %H:%M} New York: outside market hours, nothing recorded")
        return 0
    for ticker, prefix in TICKERS.items():
        try:
            board, fetched_at = fetch_cboe(ticker)
        except Exception as e:  # keep the snapshot: fall back to Yahoo
            print(f"{ticker}: Cboe failed ({type(e).__name__}: {e}), using Yahoo", file=sys.stderr)
            chains, spot, fetched_at = fetch_board(ticker)
            board = tidy(chains, spot, fetched_at)
        if not opts.force and ticker == "^SPX" and not traded_today(board, now):
            print("no SPX option traded today (holiday?): nothing recorded")
            return 0
        stamp = fetched_at.tz_convert(NY)
        path = Path(opts.out) / prefix / f"{stamp:%Y-%m-%d}" / f"{stamp:%H%M}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        board.to_csv(path, index=False, float_format="%.10g")
        print(f"{ticker} from {board['source'].iloc[0] if len(board) else '-'}: {board['expiry'].nunique()} expiries, "
              f"{len(board)} quoted contracts -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
