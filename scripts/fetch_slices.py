"""Save index option smiles for a few expiries from Yahoo into data/ (needs the network).

    .venv/Scripts/python scripts/fetch_slices.py                          # SPX, about 1 week, 1 month, 3 months
    .venv/Scripts/python scripts/fetch_slices.py 30 180                   # choose the target days
    .venv/Scripts/python scripts/fetch_slices.py 2026-10-16               # or exact expiry dates
    .venv/Scripts/python scripts/fetch_slices.py --ticker ^RUT 2026-10-16 # Russell 2000

Each file is data/<spx|rut>_<expiry>_<quote date>.csv (see marketdata.save_slice).
"""
import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from volsmile.marketdata import build_slice, fetch_chain, nearest_expiry, save_slice

parser = argparse.ArgumentParser()
parser.add_argument("targets", nargs="*", default=["7", "30", "90"], help="days to expiry or YYYY-MM-DD")
parser.add_argument("--ticker", default="^SPX", help="Yahoo index ticker, e.g. ^SPX or ^RUT")
opts = parser.parse_args()
TICKER, args = opts.ticker, opts.targets
PREFIX = TICKER.lstrip("^").lower()

ticker = yf.Ticker(TICKER)
spot = float(ticker.history(period="5d")["Close"].iloc[-1])
as_of = pd.Timestamp.now(tz="UTC")
out = Path(__file__).resolve().parent.parent / "data"
out.mkdir(exist_ok=True)
for expiry in sorted({a if "-" in a else nearest_expiry(ticker.options, as_of, float(a)) for a in args}):
    s = build_slice(*fetch_chain(TICKER, expiry), expiry, spot)
    path = out / f"{PREFIX}_{expiry}_{s.as_of:%Y-%m-%d}.csv"
    save_slice(s, path)
    q = s.quotes
    print(f"{expiry}: F {s.forward:.2f} (spot {spot:.2f}), tau {s.tau:.4f}, {len(q)} quotes, "
          f"strikes {q['strike'].min():.0f}-{q['strike'].max():.0f}, ATM-ish vol "
          f"{q.iloc[(q['strike'] - s.forward).abs().argmin()]['iv']:.4f} -> {path.name}")
