# Live recorder (`scripts/record.py`, GitHub Actions)

Whole SPX and RUT option boards from Cboe's delayed-quotes feed (Yahoo if Cboe
fails), every 15 minutes during US market hours, saved on the **`data-live`**
branch so `main` stays small.

## How it runs

- `.github/workflows/record.yml` runs on weekdays at :07, :22, :37 and :52
  from 13:07 to 21:07 UTC, which covers 09:30-16:00 New York in both summer
  and winter time. The minutes are off the quarter hours because GitHub delays
  or drops scheduled runs most at busy times (on 2026-09-14 no run on the
  quarter hours fired for 4 hours). It installs only `yfinance` and `pandas`,
  runs `record.py`, and commits any new files to `data-live`.
- `record.py` records nothing outside 09:25-16:20 New York time, and nothing
  on a market holiday (no SPX option traded today). GitHub often starts
  scheduled runs late, so snapshots are roughly, not exactly, 15 minutes
  apart; the file name is the actual fetch time.
- Each index: one Cboe request (`https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json`,
  `_RUT.json`), tried 3 times. If it still fails, the Yahoo chains are
  fetched instead (logged); a Yahoo expiry that fails 3 times is skipped, and
  if more than half fail the run fails and GitHub emails you.
- Manual run: Actions tab → *Record option boards* → *Run workflow* (tick
  *force* to record outside market hours), or
  `gh workflow run record.yml -f force=true`.

Locally:

```bash
.venv/Scripts/python scripts/record.py --force        # writes snapshots/ (gitignored)
```

## Files

    snapshots/<spx|rut>/<YYYY-MM-DD>/<HHMM>.csv.gz      New York date and time of the fetch

One row per contract with a quote (ask > 0), every listed expiry:

| Column | Meaning |
|---|---|
| `fetched_at` | fetch time, UTC ISO |
| `quote_time` | time the quotes are from, UTC ISO: Cboe feed timestamp minus 15 minutes; Yahoo: fetch time |
| `source` | `cboe` or `yahoo` |
| `spot` | index level (Cboe: the feed's `current_price`; Yahoo: latest daily close price) |
| `expiry` | YYYY-MM-DD |
| `root` | SPX / SPXW, RUT / RUTW (W = PM-settled weekly) |
| `strike`, `is_call` | contract |
| `bid`, `ask` | best quote (bid 0 = no bid) |
| `bid_size`, `ask_size` | contracts at the best bid and ask (Cboe only) |
| `volume`, `open_interest` | day volume and open interest |
| `last_trade` | contract's last trade, Unix seconds UTC |
| `cboe_iv`, `cboe_delta` | Cboe's implied vol and delta (Cboe only; their own rates and forward) |

Files recorded before 2026-09-14 are Yahoo-only and have no `quote_time`,
`source`, sizes, `cboe_iv` or `cboe_delta`.

Size at 13:26 New York on 2026-09-14: SPX 56 expiries, 28,934 contracts,
648 KB; RUT 32 expiries, 11,772 contracts, 223 KB. About 24 MB per trading
day, 500 MB a month (Yahoo was 8 MB a day).

## Why Cboe

Measured on 2026-09-14 at 13:26 New York, same moment from both sources:

- Sizes at the best bid and ask, and Cboe's own vol and delta; Yahoo has none.
- Many more two-sided quotes: SPX 3 months 266 vs 124, RUT 1 month 250 vs 90.
  Where both quote a strike, mids agree (median difference 0-0.05 in price).
- Forwards agree: SPX within 0.5, RUT within 0.4 except 3 months (1.7, on
  47 common quotes).
- One request per index, 1-3 s, against 7-14 s for Yahoo's per-expiry calls.
- Our implied vols match `cboe_iv` to a median of 0.03-0.20 vol points.

The feed is about 15 minutes behind: the latest trade in it was 2-7 seconds
before the feed timestamp minus 15 minutes. The quote time uses the feed
timestamp rather than the latest trade, which at the open would still be the
previous day's close.

## Loading

```bash
git fetch origin data-live
git worktree add ../volatility-smile-live data-live
```

```python
from volsmile.marketdata import load_snapshot
s = load_snapshot("../volatility-smile-live/snapshots/spx/2026-09-15/1037.csv.gz", "2026-10-16")
```

`load_snapshot` builds a normal slice (same contract, quote, forward and
out-of-the-money rules as `build_slice`), timed at `quote_time` (the fetch time
in older files). Sizes, `cboe_iv` and `cboe_delta` are not in the slice yet.

## Limits

- Both sources are delayed about 15 minutes; some wing quotes are stale.
- The Cboe feed is an undocumented public endpoint: it can change format or
  block cloud IP addresses. A changed symbol format raises, and the run falls
  back to Yahoo; the log says so.
- If Cboe's delay ever changes, `quote_time` is off by the difference.
  `CBOE_DELAY` in `record.py` holds it.
- On early-close days (13:00) it keeps recording until 16:20 with unchanged
  quotes.
- Git history grows by about 6 GB a year. Move old months to compressed
  archives (or Git LFS / a bucket) within a few months.
- Cboe and Yahoo data are for personal use; keep the repository private.
- The Smile Board does not read snapshots yet.
