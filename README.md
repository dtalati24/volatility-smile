# volatility-smile

An implied-volatility smile for index options (SPX and RUT), fitted to
market mids, with a fast calibrator, a live data recorder and a local web app
for marking the smile by hand.

The final design is a **least squares spline** base smile with **11 Gaussian
"bell" buttons at delta points** on top: the spline is a quick, stiff fit
through the mids, and the bells are the local adjustments a trader would
mark. **SVI** is fitted alongside only as a benchmark.

This README is the overview: design, features, what was tried and why. The
detail is in [`docs/`](docs).

---

## Contents

1. [Quick start](#quick-start)
2. [Results](#results)
3. [Design](#design)
4. [Features](#features)
5. [What we tried, and why we chose what we did](#what-we-tried-and-why-we-chose-what-we-did)
6. [Testing and review](#testing-and-review)
7. [Layout](#layout)
8. [Limitations and next steps](#limitations-and-next-steps)

---

## Quick start

```bash
py -3.14 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest -p no:warnings      # 231 tests, about 20-30 s
.venv/Scripts/python app/server.py                 # Smile Board on http://127.0.0.1:8050
```

| Command | What it does |
|---|---|
| `.venv/Scripts/python scripts/fit_slices.py` | fit every saved slice of one index (`--underlying spx\|rut`, `--knots N`), print fit errors and fit times, write `plots/fit_<underlying>.png` |
| `.venv/Scripts/python scripts/fetch_slices.py` | download new slices from Yahoo into `data/` (`--ticker ^SPX` or `^RUT`; days to expiry or dates) |
| `.venv/Scripts/python scripts/record.py --force` | record whole SPX and RUT option boards now into `snapshots/` |
| `.venv/Scripts/python explorations/<script>.py` | design experiments ([list](explorations/README.md)) |

scipy is pinned below 1.18 and matplotlib below 3.11.2: their newer extension
modules are blocked by this machine's Windows application-control policy.

---

## Results

Quotes at the close on 2026-09-11. RMSE against mid implied vols, in vol
points, spline with 3 knots.

| Slice | Quotes | Spline | Spline + bells | SVI (benchmark) |
|---|---|---|---|---|
| SPX 1 week (2026-09-21) | 156 | 0.305 | **0.173** | 0.344 |
| SPX 1 month (2026-10-16) | 203 | 0.195 | **0.110** | 0.223 |
| SPX 3 months (2026-12-18) | 118 | 0.048 | **0.035** | 0.209 |
| RUT 1 week | 79 | 0.047 | **0.044** | 0.097 |
| RUT 1 month | 78 | 0.029 | **0.026** | 0.086 |
| RUT 3 months | 46 | 1.341 | **1.311** | 1.397 |

Whole calibration time (fit the spline, then the bells), best of 40 runs on
a laptop:

| Slice | Quotes | Spline + bells, final delta | Spline + bells, spline delta |
|---|---|---|---|
| SPX 1 week | 156 | 4.0 ms | 0.5 ms |
| SPX 1 month | 203 | 2.8 ms | 0.6 ms |
| SPX 3 months | 118 | 2.3 ms | 0.5 ms |
| RUT 1 week | 79 | 1.5 ms | 0.5 ms |
| RUT 1 month | 78 | 1.7 ms | 0.5 ms |
| RUT 3 months | 46 | 2.7 ms | 0.5 ms |

The fast fitter gives the same answer as general least squares (buttons
within 0.00005 vol points).

Near-the-money bid/ask spreads on the SPX 1-month slice are 0.11-0.18 vol
points, so the spline + bells fit sits inside the spread for most strikes.
RUT spreads are about twice as wide, but its 1-week and 1-month mids are
smoother (few market makers, quoting off their own model curve); the RUT
3-month slice has stale deep put quotes 2-5 vol points off the curve, which
drive its error.

---

## Design

### 1. Market data and pricing ([docs/data-and-pricing.md](docs/data-and-pricing.md))

- **Black-76**, undiscounted: price, forward delta, vega, implied vol (root
  finding; NaN outside no-arbitrage bounds so one bad quote does not stop a
  chain).
- **A slice** is one expiry from Yahoo (yfinance):
  1. PM-settled weekly contracts (SPXW, RUTW) preferred when a date also has
     AM-settled ones, so every quote shares one expiry time.
  2. Two-sided quotes only (bid > 0, ask > bid).
  3. **Forward from put-call parity**: median of C âˆ’ P + K over the 20 strikes
     nearest spot.
  4. Out-of-the-money quotes only; implied vols recomputed from mid, bid and
     ask (Yahoo's own vols can disagree between call and put at one strike).

### 2. Spline base ([docs/calibration.md](docs/calibration.md#step-1-least-squares-spline-base-splinepy))

- Least squares **cubic spline of mid vol against k = ln(K/F)**, with
  interior knots (default **3**) at equal quantiles of the quoted strikes. A
  linear fit, under 1 ms.
- **Beyond the quotes**, total variance w = volÂ² Ï„ continues in a **straight
  line** from the spline's value and slope, with the slope kept in [0, 2]
  (never turning down; Lee's large-strike bound).

### 3. Bell overlay ([docs/models.md](docs/models.md#2-bell-overlay-bellspy))

    sigma(K) = max(0, base_vol(K) + sum_i button_i * exp(-(u - u_i)^2 / (2 w_i^2)))
    u        = -d1(F, K, tau, sigma(K))

- **11 buttons** at 1, 2, 5, 10, 25Î” puts, ATM (50Î”) and 25 â€¦ 1Î” calls. A
  button of +0.01 adds exactly 1 vol point at its node.
- **Widths** = k Ã— average distance to the neighbouring nodes (k = 0.4), so
  tip bells are narrow and middle bells wide.
- The **1Î” put and call bells hold flat past their node**, so the far wings
  carry on at the lifted level instead of dropping back onto the spline.
- **Bells at the final smile's delta** (default): u uses the finished vol,
  so every node sits at its true delta. Each strike's vol is therefore solved
  per strike: a cheap bound proves most strikes have exactly one solution
  (86-100% on the saved slices); the rest are scanned over the range the
  bells can reach. Then safeguarded Newton. More than one solution raises
  `AmbiguousVolError`.
  The alternative, **bells at the spline's delta**, is explicit and always
  unique.

### 4. Calibration ([docs/calibration.md](docs/calibration.md))

Objective, with the spline fixed:

    minimise  sum over quotes (model vol - mid)^2  +  stiffness * sum over buttons (button - start)^2

- **Fast solver** (default):
  - *Spline delta*: the bells are linear in the buttons, so one bounded
    linear least squares gives the exact minimum.
  - *Final delta*: **Gauss-Newton with an analytic Jacobian**. Raising a
    button raises the vol, which moves the strike's delta along every bell,
    so the sensitivity is
    `d vol_i / d button_j = bell_j(u_i) / (1 - g_i)`,
    `g_i = (bell slopes . buttons) * d2 / vol`.
    Vols between steps come from Newton started at the linear prediction
    (vol + J Ã— step); the final answer is checked with the full solve.
  - Hands over to the general solver, with the reason recorded, when its
    maths does not apply (vol floor binds, ambiguous vol, g â‰¥ 1, no
    convergence).
- **General solver**: `scipy.optimize.least_squares` with finite differences,
  kept as the reference and the fallback.
- Buttons no quote can see (bell below 0.001 at every quote) are held at
  their start; otherwise noise sends them to their Â±20 vol point bounds.
- **SVI** (raw parametrisation, 8 starts, Lee wing bound) is fitted to the
  same mids as a benchmark only.

### 5. Automatic fitter and stale spline

- **Refit**: fresh spline + bells, fitted to the mids.
- **Automatic fitter**: spline kept (tied to moneyness, so it moves with the
  forward), bells refitted from their previous values, optionally with
  stiffness so noisy quotes move them less.
- **Offsets**: a trader's manual clicks sit on top of the fitted bells and
  are never undone by the automatic fitter; Refit clears them.
- **Stale-spline alert** when the bells' fit error exceeds 1.5Ã— the error
  after the last Refit, a fitted button passes 2 vol points, or a vol becomes
  ambiguous.

### 6. Live recorder ([docs/recorder.md](docs/recorder.md))

- GitHub Actions records the **whole SPX and RUT boards every 15 minutes**
  in US market hours from Cboe's delayed-quotes feed (Yahoo as a fallback):
  about 40,000 contracts, ~870 KB per snapshot, with **sizes at the best bid
  and ask** and Cboe's own vol and delta, to the `data-live` branch. Skips
  holidays (no SPX option traded that day).
- `marketdata.load_snapshot(path, expiry)` turns any snapshot into a slice.

---

## Features

**Smile Board** ([docs/app.md](docs/app.md)): local web app, standard library
server, no front-end libraries.

- Smile chart: market mids with bid/ask bars, spline base (blue dashed),
  spline + bells (orange), SVI (green, optional), forward line, node ticks,
  hover tooltips; zoom buttons and drag to pan.
- Option board: bid, ask, mid, model price, model delta, bid/mid/ask vols,
  model vol, model âˆ’ mid, volume, open interest; model vols outside the
  bid/ask shaded.
- 11 bell buttons: click raises your offset, Ctrl + click lowers it; each
  shows fitted + offset.
- Controls: slice (SPX or RUT, any saved expiry), knots, bell width k, delta
  mode, click step, Refit, Auto-fit bells, Auto, Stiffness, Clear offsets,
  Zero buttons, full strike range, show SVI.
- Stats line with fit errors and the last fit's time and solver; stale-spline
  banner; errors (e.g. ambiguous vols) shown and the click reverted.
- JSON API: `/api/slices`, `/api/fit`, `/api/autofit`, `/api/smile`.
- Hardened: bound to 127.0.0.1, Host header check (DNS rebinding), body size
  cap, no NaN in JSON, every error answered.

**Data**: saved SPX and RUT slices in `data/`, fetch script for new ones,
whole-board recorder on GitHub Actions.

**Scripts**: fit every slice with errors, timings and plots; fetch and record
data; bell design explorations.

---

## What we tried, and why we chose what we did

### The bell model

Each button means "vol at this delta": a trader's natural way to mark a smile.

| Tried | Outcome and reasoning |
|---|---|
| One fixed bell width | rejected: narrow enough for the tips leaves holes in the middle; wide (0.6) is ill-conditioned (condition number 616) |
| Width = k Ã— neighbour distance, k = 0.4 | **adopted** (condition number 1.3) |
| Standalone: ATM vol + bells, straight line past 1Î” | replaced: the bells are meant to adjust an existing smile, not rebuild it |
| **Overlay on a base smile** | **adopted** |
| Bells at deltas from ATM vol | rejected: on a steep put wing a "1Î”" node at ATM vol can sit on a ~13Î” put |
| **Bells at the final smile's delta** | **adopted** (chosen: nodes at true deltas, regions barely move); spline delta kept as an option |
| Repeating vol â†’ delta â†’ vol to solve it | rejected: does not settle from Â±0.75 vol point alternating buttons; replaced by bracketing + safeguarded Newton |
| Last bells fading past 1Î” | replaced: a +0.41 vol point 1Î” call bell dropped the curve back onto a spline 0.4-0.7 vol points under the far calls |
| **1Î” bells held flat past their node** | **adopted**: same RMSE, far-call misses cut from âˆ’0.66/âˆ’0.61/âˆ’0.44 to âˆ’0.42/âˆ’0.36/âˆ’0.19 |

### The base smile

| Tried | Outcome and reasoning |
|---|---|
| SVI as the base | built first, then **reversed**: the bells are meant to replace SVI, not sit on top of it |
| **Least squares spline** | **adopted**: simple, linear, fast, beats SVI on every slice |
| 2, 3, 4, 6, 20 knots | **3**: from 6 knots the spline alone follows the mids and the bells add almost nothing |
| x-axis strike, delta, ln(K/F) | **ln(K/F)**: smoother than strike; unlike delta it does not depend on the vol being fitted |
| Cubic extrapolation | rejected: shoots off; straight-line total variance with Lee-bounded slope instead |
| SVI without a wing bound | fitted b = 10, rho = 0.99 on the 1-week slice, extrapolating to 89% vol; bound added |

Why SVI fits worse: five parameters and one fixed hyperbola shape. On the
3-month slice its misses swing +0.12, âˆ’0.14, âˆ’0.15, +0.02, +0.20, âˆ’0.06 vol
points across the strikes while the spline's are noise around 0.

### The fitter

| Tried | Outcome and reasoning |
|---|---|
| General least squares (scipy, finite differences) | worked, 1-2 s per SPX fit; kept as reference and fallback |
| Frozen-delta linear re-solves (fixed point) | rejected: same RMSE but buttons up to 0.08 vol points off the true least squares minimum |
| **Linear solve / Gauss-Newton with analytic Jacobian** | **adopted**: same answer as scipy, ~20 ms at first |
| Profiling that fit | over half the time was the uniqueness scan (257 trial vols per strike), the rest numpy call overhead |
| **Proving uniqueness with a bound** (the gap's slope stays below 0 across the reachable vols) | **adopted**: 86-100% of strikes skip the scan; identical vols and ambiguity results on random cases |
| **Less overhead** (cached widths, one exponential for heights and slopes, no re-validation inside the solver) and **Newton started from the linear prediction** | **adopted**: with the bound, 2-4 ms per SPX fit, same buttons |
| Tighter per-strike vol ranges in numpy | rejected: the extra work cost more than the scans it saved |
| Looser convergence tolerance | rejected: saved at most one step, no measurable gain |
| Compiled kernels (numba) | tested at 0.3-2 ms per fit with identical buttons; not adopted for now: a new dependency, a compile delay per process, and a second implementation to keep in step |
| Step tolerance 1e-9 | too tight: below the objective's rounding noise, a converged fit looked stuck; 1e-7 |
| Newton accepting any in-bracket step | a review found it could bounce between two vols and return a non-root; rtsafe halving rule + bisection safety net |
| Fitting every button exactly | buttons no quote can see went to Â±20 vol points on noise; held at their start |
| Offsets re-fitted "around" by the automatic fitter | rejected: the same button's fitted part would cancel the click; offsets sit on top instead |
| Automatic spline refit when stale | not chosen: an alert instead, so the trader decides |

### Data

| Tried | Outcome and reasoning |
|---|---|
| Yahoo's implied vols | not used: call and put disagree at one strike |
| Nearest expiry to 30 days | a thin Tuesday weekly (41 quotes); the 2026-10-16 monthly (203) used instead |
| Euro Stoxx 50 options | not on Yahoo (^STOXX50E, SX5E, EXW1.DE empty) |
| FEZ (Euro Stoxx ETF) | declined: American exercise |
| NDX, XSP, DJX, XEO, MRUT | too few quotes on Yahoo |
| **RUT** | **added**: European exercise, cash-settled, rougher quotes |
| Free historical intraday option quotes | none exist; **record our own** every 15 minutes |
| Recording every 1 minute | declined: GitHub schedules at most every 5 minutes, ~8,200 billed minutes a month, Yahoo blocking risk, ~2.5 GB a month |
| **Cboe delayed-quotes feed** for the recorder | **adopted** (Yahoo kept as fallback): sizes at the best bid and ask, up to 2.8x the two-sided quotes per expiry, one request per index; same mids and forwards where both quote |
| Scheduled runs on the quarter hours | moved to :07/:22/:37/:52: on the first trading day none of them fired for 4 hours |

### App

| Tried | Outcome and reasoning |
|---|---|
| `click` / `contextmenu` events for buttons | double-counted Mac Ctrl + click; read on mouse-down instead |
| Each click starting from the last server reply | quick clicks were lost; clicks now build on the value shown |
| Browser caching of the page script | an old script hid new buttons; static files now sent with no-cache |
| A quick auto-fit unlocking buttons mid-Refit | found in review; fits are counted and buttons stay locked until all finish |

---

## Testing and review

- **231 tests** (pytest): hand-computed values, independent solvers (brentq)
  for per-strike vols, finite differences for the Jacobian, fast vs general
  fitter on every saved slice, recovery of known buttons, every hand-over
  path, the web API end to end.
- **Mutation checks**: deliberately broken copies of key lines (the 1 / (1 âˆ’ g)
  term, held buttons, stiffness sign, ambiguity checks, warm start, the Newton
  halving rule, offsets in drawing, â€¦) to confirm a test fails for each.
- **Independent review** of every stage (a separate agent running the code);
  every finding was fixed or documented as a known limitation.

---

## Layout

```
src/volsmile/
  black76.py       Black-76 price, delta, vega, implied vol (undiscounted)
  marketdata.py    SPX / RUT slices and recorded snapshots
  spline.py        least squares spline base with straight-line wings
  bells.py         bell overlay; per-strike vol solve; sensitivities
  svi.py           raw SVI and its fit (benchmark)
  calibration.py   spline fit, fast and general bell fits, stale-spline check
app/               Smile Board: server.py + static page
scripts/           fetch, fit, record
data/              saved SPX and RUT slices (CSV)
explorations/      design experiments kept for reference
docs/              reference documentation
tests/             pytest suite
.github/workflows/ recorder schedule
```

| Document | Covers |
|---|---|
| [docs/models.md](docs/models.md) | spline, SVI, bell overlay (nodes, widths, deltas, ambiguity, arbitrage) |
| [docs/calibration.md](docs/calibration.md) | spline fit, bell objective, fast and general solvers, speed, automatic fitter, results |
| [docs/data-and-pricing.md](docs/data-and-pricing.md) | Black-76, building slices, saved snapshots, data limitations |
| [docs/app.md](docs/app.md) | Smile Board controls, offsets, API |
| [docs/recorder.md](docs/recorder.md) | 15-minute board recorder, file format, loading |

---

## Limitations and next steps

- **Spot dynamics**: the spline is tied to moneyness for now; how
  the smile and bells should move with spot is to be measured on the recorded
  snapshots.
- **Snapshot replay** in the Smile Board (time slider), driving the automatic
  fitter; a stability study of the buttons through the day.
- **Arbitrage checks**: butterfly and calendar; narrow bells can create
  butterfly arbitrage (5Î” put +1 vol point at k = 0.4).
- **Weights** by spread, volume or open interest (fits are unweighted).
- **More nodes** beyond 1Î”; better spline wings (the 9100 SPX call is ~1 vol
  point off).
- **Discounting** (about 0.02-0.07 vol points at 4% rates) and 09:30 expiry
  for AM-settled contracts.
- **Full surface** across expiries.
- Yahoo and Cboe option quotes are delayed about 15 minutes; data is for
  personal use. Sizes are recorded but not used in fits yet.
