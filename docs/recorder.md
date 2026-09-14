# Live recorder (`scripts/record.py`, GitHub Actions)

Whole SPX and RUT option boards from Yahoo, every 15 minutes during US market
hours, saved on the **`data-live`** branch so `main` stays small.

## How it runs

- `.github/workflows/record.yml` runs on weekdays every 15 minutes from 13:00
  to 21:15 UTC, which covers 09:30-16:00 New York in both summer and winter
  time. It installs only `yfinance` and `pandas`, runs `record.py`, and
  commits any new files to `data-live`.
- `record.py` records nothing outside 09:25-16:20 New York time, and nothing
  on a market holiday (no SPX option traded today). GitHub often starts
  scheduled runs 5-15 minutes late, so snapshots are roughly, not exactly,
  15 minutes apart; the file name is the actual fetch time.
- Manual run: Actions tab → *Record option boards* → *Run workflow* (tick
  *force* to record outside market hours), or
  `gh workflow run record.yml -f force=true`.
- An expiry that fails 3 times is skipped (logged); if more than half fail
  the run fails and GitHub emails you.

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
| `spot` | index level at fetch |
| `expiry` | YYYY-MM-DD |
| `root` | SPX / SPXW, RUT / RUTW (W = PM-settled weekly) |
| `strike`, `is_call` | contract |
| `bid`, `ask` | quote (bid 0 = no bid) |
| `volume`, `open_interest` | Yahoo's day volume and open interest |
| `last_trade` | contract's last trade, Unix seconds UTC |

Size on 2026-09-11's close: SPX 51 expiries, 16,682 contracts, 253 KB; RUT 27
expiries, 3,809 contracts, 53 KB. About 8 MB per trading day, 170 MB a month.

## Loading

```bash
git fetch origin data-live
git worktree add ../volatility-smile-live data-live
```

```python
from volsmile.marketdata import load_snapshot
s = load_snapshot("../volatility-smile-live/snapshots/spx/2026-09-14/1030.csv.gz", "2026-10-16")
```

`load_snapshot` builds a normal slice (same contract, quote, forward and
out-of-the-money rules as `build_slice`), with the fetch time as the quote
time. Checked: a snapshot recorded after the 2026-09-11 close loads to the
same quotes and forwards as the saved `data/` slices for all six expiries.

## Limits

- Yahoo option quotes are delayed (typically about 15 minutes) and some wing
  quotes are stale; the fetch time is used as the quote time.
- On early-close days (13:00) it keeps recording until 16:20 with unchanged
  quotes.
- Yahoo can block or rate-limit cloud IP addresses; if runs start failing,
  that is the first suspect.
- Git history grows by ~2 GB a year. Move old months to compressed archives
  (or Git LFS / a bucket) before that matters.
- Yahoo data is for personal use; keep the repository private.
- The Smile Board does not read snapshots yet.
