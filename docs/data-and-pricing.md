# Pricing and market data

## Black-76 (`black76.py`)

European options on a forward, **undiscounted** (discount factor 1):

    d1 = ln(F/K) / (vol sqrt(tau)) + vol sqrt(tau) / 2,   d2 = d1 - vol sqrt(tau)
    call = F N(d1) - K N(d2),   put = K N(-d2) - F N(-d1)

| Function | Returns |
|---|---|
| `price(F, K, tau, vol, is_call)` | option price; vol 0 gives intrinsic value |
| `d1(F, K, tau, vol)` | d1 |
| `delta(F, K, tau, vol, is_call)` | forward delta: `N(d1)` calls, `-N(-d1)` puts |
| `vega(F, K, tau, vol)` | `F phi(d1) sqrt(tau)`, per 1.00 of vol |
| `implied_vol(price, F, K, tau, is_call)` | vol strictly between 1e-4 and 5 that reproduces the price, else NaN |

All accept arrays and broadcast. Bad `F`, `K`, `tau` or `vol` raise
`ValueError`. `implied_vol` returns NaN (rather than raising) for prices at or
outside the no-arbitrage bounds, so one dirty quote does not stop a chain.

## SPX and RUT slices (`marketdata.py`)

A **slice** is one expiry: out-of-the-money quotes with mid prices and implied
vols, plus the forward and time to expiry.

```python
from volsmile.marketdata import load_slice
s = load_slice("data/spx_2026-10-16_2026-09-11.csv")
s.forward, s.tau, s.quotes.head()
```

`quotes` columns: `strike, is_call, bid, ask, mid, iv, iv_bid, iv_ask, delta,
volume, open_interest`.

### How a slice is built (`build_slice`)

1. **Contracts.** When a date has two-sided quotes on both the PM-settled
   weekly root (SPXW, RUTW: the root ending in W) and the AM-settled one (SPX,
   RUT), only the PM root is kept, so each strike appears once and every quote
   shares one expiry time. (The check runs after the quote filter, so a PM root
   listed without quotes does not hide the AM quotes.)
2. **Quotes.** Kept if `bid > 0` and `ask > bid`.
3. **Forward.** Put-call parity, `C - P = F - K`: the median of `C - P + K`
   over the 20 strikes nearest spot that have two-sided quotes on both sides.
4. **Time.** From the chain's latest trade time (UTC) to 16:00 New York time
   on the expiry date, in calendar days / 365.
5. **Out of the money only.** Puts with `K < F`, calls with `K >= F`.
6. **Vols.** Implied vols from mid, bid and ask against that one forward
   (Yahoo's own `impliedVolatility` can disagree between calls and puts at the
   same strike by several vol points). Quotes whose mid has no implied vol are
   dropped. Delta is the forward delta at the mid vol.

### Saved snapshots

`scripts/fetch_slices.py` (`--ticker ^SPX` or `^RUT`) downloads from Yahoo and writes
`data/<spx|rut>_<expiry>_<quote date>.csv`. Snapshots in the repo (quotes from the
close on Friday 2026-09-11, spot 7656.98):

| File | Expiry | tau | Forward | Quotes |
|---|---|---|---|---|
| `spx_2026-09-21_2026-09-11.csv` | 1 week | 0.0274 | 7661.25 | 156 |
| `spx_2026-10-16_2026-09-11.csv` | 1 month (monthly) | 0.0959 | 7682.17 | 203 |
| `spx_2026-12-18_2026-09-11.csv` | 3 months | 0.2686 | 7728.65 | 118 |

RUT (Russell 2000, spot 2903.94, same close):

| File | Expiry | tau | Forward | Quotes |
|---|---|---|---|---|
| `rut_2026-09-21_2026-09-11.csv` | 1 week (RUTW) | 0.0274 | 2906.85 | 79 |
| `rut_2026-10-16_2026-09-11.csv` | 1 month (RUTW; the AM RUT monthly has more strikes, down to 1750) | 0.0959 | 2913.40 | 78 |
| `rut_2026-12-18_2026-09-11.csv` | 3 months (AM-settled RUT only) | 0.2686 | 2929.77 | 46 |

Sanity check on the one-month slice: call and put implied vols at the same
strike within 3% of the forward agree to within about 0.1 vol points (0.22 at
the edge of that band), and near-the-money bid/ask spreads are 0.11–0.18 vol
points.

### Known limitations

- **No discounting.** A review estimated that at 4% rates this moves the
  forward by +0.06 points (1 month) / +0.17 points (3 months) and lowers vols
  by about 0.02 / 0.07 vol points (median), largest at the money. Small, but
  comparable to the fit error after the bells. Fix when needed: regress `C - P`
  on `K` near spot to get the discount factor as well as the forward.
- **Quote time.** Taken from the latest trade, while bid/ask are a snapshot at
  fetch time. Fine for the saved files (fetched after the close); a thin
  expiry or an overnight fetch could misdate it. Recorded snapshots use their
  own quote time instead ([recorder.md](recorder.md)).
- **AM-settled-only dates** still get a 16:00 expiry (settlement is at the
  morning open), which matters only for very short expiries.
- Yahoo data is for personal use; check its terms before publishing the CSVs.
