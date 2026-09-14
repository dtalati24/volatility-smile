"""Record whole SPX and RUT option boards from Yahoo (needs the network).

    python scripts/record.py                      # during US market hours only
    python scripts/record.py --force              # record now, whatever the time
    python scripts/record.py --out live/snapshots

Writes <out>/<spx|rut>/<YYYY-MM-DD>/<HHMM>.csv.gz, named by New York time, one
row per contract with a quote (ask > 0) on every listed expiry. Columns are
SNAPSHOT_COLUMNS; `volsmile.marketdata.load_snapshot` turns one expiry of a file
into a slice. Run every 15 minutes by .github/workflows/record.yml, which saves
the files on the data-live branch.

Outside 09:25-16:20 New York time on weekdays it records nothing (the workflow
runs on a UTC schedule wide enough for both summer and winter time, and GitHub
often starts scheduled runs late). On a market holiday it records nothing
either: no SPX option traded today.

Only pandas and yfinance are needed, not the volsmile package, so the workflow
installs little.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, time as clock
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

TICKERS = {"^SPX": "spx", "^RUT": "rut"}
NY = ZoneInfo("America/New_York")
OPEN, CLOSE = clock(9, 25), clock(16, 20)
SNAPSHOT_COLUMNS = ["fetched_at", "spot", "expiry", "root", "strike", "is_call", "bid", "ask",
                    "volume", "open_interest", "last_trade"]
RETRIES = 3


def market_hours(now: datetime) -> bool:
    """Weekday, between OPEN and CLOSE New York time."""
    ny = now.astimezone(NY)
    return ny.weekday() < 5 and OPEN <= ny.time() <= CLOSE


def tidy(chains: dict[str, tuple[pd.DataFrame, pd.DataFrame]], spot: float, fetched_at: pd.Timestamp) -> pd.DataFrame:
    """One row per quoted contract, in SNAPSHOT_COLUMNS. last_trade is Unix seconds (compact)."""
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
                # Not astype(int64): its unit depends on the pandas datetime resolution.
                "last_trade": (pd.to_datetime(df["lastTradeDate"], utc=True) - pd.Timestamp(0, tz="UTC"))
                              // pd.Timedelta(seconds=1),
            }))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SNAPSHOT_COLUMNS[2:])
    out.insert(0, "spot", spot)
    out.insert(0, "fetched_at", fetched_at.isoformat())
    return out[SNAPSHOT_COLUMNS]


def fetch_board(ticker: str) -> tuple[dict, float, pd.Timestamp]:
    """Every expiry's chain. An expiry that still fails after RETRIES tries is skipped."""
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
        chains, spot, fetched_at = fetch_board(ticker)
        board = tidy(chains, spot, fetched_at)
        if not opts.force and ticker == "^SPX" and not traded_today(board, now):
            print("no SPX option traded today (holiday?): nothing recorded")
            return 0
        stamp = fetched_at.tz_convert(NY)
        path = Path(opts.out) / prefix / f"{stamp:%Y-%m-%d}" / f"{stamp:%H%M}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        board.to_csv(path, index=False, float_format="%.10g")
        print(f"{ticker}: {len(chains)} expiries, {len(board)} quoted contracts, spot {spot:.2f} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
