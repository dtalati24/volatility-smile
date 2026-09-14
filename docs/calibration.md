# Calibration (`spline.py`, `calibration.py`, `svi.py`)

Two steps, both least squares on implied vol against **market mids**,
unweighted. SVI is fitted to the same mids as a **comparison only**. Weighting
by open interest, volume or bid/ask spread is left for later.

```python
from volsmile.marketdata import load_slice
from volsmile.calibration import fit_slice
fit = fit_slice(load_slice("data/spx_2026-10-16_2026-09-11.csv"), knots=3)
fit.base, fit.buttons, fit.rmse_base, fit.rmse_total, fit.rmse_svi, fit.bells_converged, fit.bells_method
```

## Step 1: least squares spline base (`spline.py`)

**What is fitted.** Implied vol at the mid (y) against log-moneyness
k = ln(K/F) (x).

- **Cubic spline:** cubic pieces joined at `knots` interior knots, with value,
  slope and curvature matching at each knot.
- **Knots** (default **3**, adjustable): placed at equal quantiles of the
  distinct quoted k (3 knots: the 25%, 50% and 75% points), so each piece
  holds about the same number of strikes and the knots land where quotes are
  dense.
- **Least squares:** the curve minimises the total squared miss in vol rather
  than passing through every mid, which smooths quote noise. It is a linear
  fit (`scipy.interpolate.make_lsq_spline`), so it takes under 3 ms.

**Why ln(K/F).** The smile is smoother in log strike than in strike; unlike
delta it does not depend on the vol being fitted; the wing rule and the Lee
bound below are stated in it. (Dividing by ATM vol × √τ would give the same
fit for one expiry.)

**Beyond the quotes.** A cubic shoots off quickly outside its data, so past the
lowest and highest quoted strike:

1. switch to total variance w = vol² × τ,
2. take the spline's value and slope of w at the last quote,
3. continue w in a straight line in k.

So there is no jump and no kink at the last quote, and vol keeps rising more
and more slowly (like a square root) further out. The outward slope is kept
between 0 (the wing never turns down) and 2 (Lee's large-strike no-arbitrage
bound: total variance can grow at most like 2|k|). A slope clipped to a limit
leaves a kink at the last quote. Inside the quotes the spline vol is floored
at 0.

**Choosing knots.** More knots follow the market more closely and leave less
for the bells. Fit error in vol points, spline alone → spline + fitted bells:

| Knots | 1 week | 1 month | 3 months |
|---|---|---|---|
| 2 | 0.367 → 0.215 | 0.326 → 0.206 | 0.202 → 0.128 |
| **3 (default)** | **0.305 → 0.173** | **0.195 → 0.110** | **0.048 → 0.035** |
| 4 | 0.126 → 0.081 | 0.064 → 0.055 | 0.035 → 0.021 |
| 6 | 0.062 → 0.058 | 0.063 → 0.059 | 0.032 → 0.019 |
| 20 | 0.048 → 0.048 | 0.043 → 0.043 | 0.007 → 0.007 |

With 6 or more knots the bells add almost nothing. 3 keeps the base stiff so
the bells carry the local shape.

## Step 2: bells on top (`fit_bells`)

**Objective.** With the spline base fixed:

    minimise  sum over quotes (model vol - mid)^2  +  stiffness * sum over buttons (button - start)^2

- Every button within ±0.2 (20 vol points).
- `start` is where the solver starts (0 for a fresh fit); `stiffness` (default
  0) pulls the buttons towards it. Stiffness is for the automatic fitter: with
  1, moving a button 0.1 vol points costs as much as one quote missing by 0.1
  vol points, so one noisy quote moves the buttons less.
- A button whose bell is below 0.001 at every quote (`MIN_BELL`) is held at
  its start: the quotes cannot pin it down, and fitting it anyway sends it to
  ±20 vol points on noise (a test shows this for a 2Δ put bell of height 4e-7).
- The 1Δ buttons also move every quote beyond their node (they hold flat out
  to the wings).

```python
from volsmile.calibration import fit_bells
b = fit_bells(slice, base_vol, cfg, method="fast", start=None, stiffness=0.0)
b.buttons, b.converged, b.method, b.handover
```

### Fast solver (default)

**Base delta: one linear solve.** Each bell sits at the base smile's delta,
which the buttons do not change, so

    vol = base + bell_matrix(u) @ buttons        (u fixed)

is linear in the buttons: one bounded linear least squares gives the exact
minimum (`numpy.linalg.lstsq`, or `scipy.optimize.lsq_linear` if a bound
binds).

**Final delta: Gauss-Newton with exact sensitivities.** A button changes a
strike's vol, which changes its delta, which slides it along every bell. The
sensitivity including that knock-on is exact (`bells.vol_sensitivity`):

    d vol_i / d button_j = bell_j(u_i) / (1 - g_i)
    g_i = (bell slopes at u_i . buttons) * d u / d vol,   d u / d vol = d2 / vol

1. Start from the base-delta answer (or `start` when given, e.g. the previous
   fit).
2. Solve every strike's vol for the current buttons: Newton from the last
   vols (`bells.strike_vol_near`, no ambiguity scan).
3. Build the sensitivities; take the bounded linear least squares step for the
   objective above; halve it until the objective does not rise.
4. Repeat until no button would move more than 1e-7 (0.00001 vol points).
   Smaller steps change the objective by less than its rounding noise.
5. Check the answer with the full scan (`bells.strike_vol`): every strike must
   have exactly one solution, equal to the one followed.

**Hand-overs.** It hands over to the scipy solver, and `BellFit.handover` says
why, when its maths does not apply:

| Reason | What happened |
|---|---|
| `vol floor at 0 binds` | the fitted bells would push a vol below 0 (base delta), or a vol sits on the floor |
| `ambiguous vol` | a strike has more than one self-consistent vol for the buttons |
| `vol sensitivity blows up (g >= 1)` | a strike is at the edge of having two solutions |
| `objective would not fall` | halving the step never improved the objective |
| `did not converge` | 30 steps without meeting the step test |

On the six saved slices, with 3 and 6 knots and both delta modes (24 fits), it
never hands over, and it matches the scipy solver to within 0.00005 vol points
per button with the same RMSE (test: `tests/test_fast_fit.py`).

### Scipy solver (`method="scipy"`, and the hand-over target)

`scipy.optimize.least_squares` on the same objective with finite differences:
first in base-delta mode, then (final delta) refined from there. At most 300
evaluations per run. Button values that make a strike's vol ambiguous are
scored as a 1.0 vol error per quote, which is a cliff, not a slope: a fit that
ends right next to the ambiguous region can stop early (a review saw this with
targets built from ±2 vol point alternating buttons). `BellFit.converged` shows
it.

### Speed

Whole calibration (spline fit + bell fit), 3 knots, best of 40 runs on a
laptop (timings on it vary by up to 2x between runs):

| Slice | Quotes | Spline + bells, final delta | Spline + bells, spline delta |
|---|---|---|---|
| SPX 1 week | 156 | 4.0 ms | 0.5 ms |
| SPX 1 month | 203 | 2.8 ms | 0.6 ms |
| SPX 3 months | 118 | 2.3 ms | 0.5 ms |
| RUT 1 week | 79 | 1.5 ms | 0.5 ms |
| RUT 1 month | 78 | 1.7 ms | 0.5 ms |
| RUT 3 months | 46 | 2.7 ms | 0.5 ms |

What makes the final-delta fit fast, measured on the SPX 1-month slice:

1. **Gauss-Newton with the analytic Jacobian** instead of finite differences
   (see above): about 1.7 s -> 20 ms.
2. **No scan for strikes with provably one solution.** Profiling showed over
   half of those 20 ms in the final uniqueness check, a 257-vol scan per
   strike. `bells._one_solution` bounds the gap's slope over every vol a
   strike can reach,

       |g| <= sum_j |button_j| * max |bell_j slope| over the reachable deltas * max |d2 / vol|,

   and where that is below 1 the gap falls all the way, so there is exactly
   one solution. 86-100% of strikes on the saved slices qualify; only the
   rest are scanned. Tests check it against dense scans and against scanning
   every strike (same vols and ambiguity results).
3. **Less numpy overhead:** bell widths cached, one exponential for bell
   heights and slopes, no input re-validation inside the solver loops.
4. **Newton started from the linear prediction** vol + J × step between
   Gauss-Newton steps, so each per-strike solve needs fewer iterations.

Tried and not used: tighter per-strike vol ranges (in numpy the extra work
cost more than it saved), a looser convergence tolerance (at most one step
saved), and compiled numba kernels (0.3-2 ms per fit with identical buttons,
but a new dependency, a compile delay per process and a second
implementation to maintain). `scripts/fit_slices.py` prints the fast and
scipy bell fit times for each saved slice.

### Automatic fitter and stale spline

The Smile Board's automatic fitter keeps the spline (tied to moneyness, so it
moves with the forward) and refits only the bells, from the previous fitted
buttons, optionally with stiffness. `calibration.stale_reasons` flags the
spline as stale when:

- the bells' fit error is above 1.5× the error just after the last full fit
  (`STALE_ERROR_RATIO`),
- any fitted button is beyond 2 vol points (`STALE_BUTTON`), or
- the fast solver met an ambiguous vol.

A whole-smile move the bells cannot make (a test shifts every mid by +3 vol
points) trips it; 0.02 vol point noise does not.

## Comparison: SVI (`svi.py`)

Raw SVI, `w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))`, fitted to
the same mids from 8 starting points. Reparametrised so box bounds keep the
minimum variance ≥ 0 and the wing slopes b(1 + |rho|) ≤ 2 (Lee). Without the
wing bound the one-week slice fitted b = 10, rho = 0.99 with its vertex outside
the quotes, extrapolating to 89% vol just past the last call. If SVI does not
converge, `Fit.svi` and `Fit.rmse_svi` are `None`; the rest of the fit is
unaffected.

## Results on the saved snapshots (3 knots)

`scripts/fit_slices.py` (optionally `--underlying rut`, `--knots N`) prints this table and writes
`plots/fit_<underlying>.png`: market mids with bid/ask bars, the spline base,
spline + bells, SVI, and the errors.

| Expiry | Quotes | Spline | Spline + bells | SVI (comparison) |
|---|---|---|---|---|
| 2026-09-21 (1 week) | 156 | 0.305 | 0.173 | 0.344 |
| 2026-10-16 (1 month) | 203 | 0.195 | 0.110 | 0.223 |
| 2026-12-18 (3 months) | 118 | 0.048 | 0.035 | 0.209 |

RUT (Russell 2000, European exercise), same quote time:

| Expiry | Quotes | Spline | Spline + bells | SVI (comparison) |
|---|---|---|---|---|
| 2026-09-21 (1 week) | 79 | 0.047 | 0.044 | 0.097 |
| 2026-10-16 (1 month) | 78 | 0.029 | 0.026 | 0.086 |
| 2026-12-18 (3 months) | 46 | 1.341 | 1.311 | 1.397 |

RUT bid/ask spreads are about twice SPX's (one month, near the money: 0.30 vs
0.16 vol points), yet the one-week and one-month RUT mids are *smoother*:
fewer market makers, quoting both sides off their own smooth vol surface, so
the mid is close to a model curve. The three-month RUT slice is only
AM-settled contracts with stale deep put quotes 2-5 vol points off the curve,
which drive its 1.3 vol point error.

All SPX fits converged; every fitted button is under 0.6 vol points in size. The
remaining error is largest at the deepest wing quotes, beyond the 1Δ nodes,
where only the base shifted by the 1Δ button applies. Near-the-money bid/ask spreads on the one-month
slice are 0.11–0.18 vol points.

## Known limitations

- A button that is only weakly identified (a few quotes at the edge of its
  bell) has nothing pulling it back to 0. A small ridge penalty would fix that
  if it shows up.
- No arbitrage checks on the fitted smile (left out on purpose for now). A
  flexible spline, or narrow bells with large buttons, can create butterfly
  arbitrage.
