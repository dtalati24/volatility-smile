# Smile models

| Module | What it is | Used by the fit and the app |
|---|---|---|
| `spline.py` | **Least squares spline**: the base smile, fitted to market mids | yes, as the base |
| `bells.py` | **Bell overlay**: eleven buttons at delta nodes, added on top of any base smile | yes, on top of the spline |
| `svi.py` | **SVI**: fitted to the same mids | comparison only |

Units everywhere: vols and buttons in decimals (0.01 = 1 vol point), time in
years, strikes and forwards in index points.

---

## 1. Spline base (`spline.py`) and SVI comparison (`svi.py`)

**Spline.** A least squares cubic spline of vol against k = ln(K/F), with 3
interior knots by default (adjustable) at equal quantiles of the distinct quoted strikes. Past
the quotes, total variance continues in a straight line with the spline's
value and slope at the last quote, slope kept between 0 and 2.

**SVI**, for comparison:

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))    total variance
    vol  = sqrt(w / tau)

Both fits are described in [calibration.md](calibration.md).

---

## 2. Bell overlay (`bells.py`)

### Formula

    sigma(K) = max(0, base_vol(K) + sum_i button_i * exp(-(u - u_i)^2 / (2 w_i^2)))
    u        = -d1(F, K, tau, sigma(K))          Black-76 d1 at the strike's final vol

With forward delta, a put's delta is `-N(u)` and a call's is `N(-u)`, so a
node at `u_i` sits exactly at a delta.

### Nodes and widths (k = 0.4)

| Button | Delta | u_i | Width w_i |
|---|---|---|---|
| `put_1` | 1Δ put | -2.3263 | 0.1090 |
| `put_2` | 2Δ put | -2.0537 | 0.1363 |
| `put_5` | 5Δ put | -1.6449 | 0.1544 |
| `put_10` | 10Δ put | -1.2816 | 0.1941 |
| `put_25` | 25Δ put | -0.6745 | 0.2563 |
| `atm` | 50Δ (d1 = 0, just above F) | 0 | 0.2698 |
| `call_25` … `call_1` | mirror of the puts | +0.6745 … +2.3263 | same |

- `u_i = N^-1(delta)` for puts, `-N^-1(delta)` for calls.
- The nodes bunch up towards the tips on purpose, for finer control there.
- Width = `k` × average distance to the neighbouring nodes (one-sided at the
  tips), so tip bells are narrow and middle bells wide. `k` is adjustable
  (`BellConfig(k=...)`); 0.4 was chosen from `explorations/delta_bells.py`.
- A button of +0.01 adds exactly 1 vol point at its own node and a little at
  its neighbours. The inner bells fade out away from their node.
- **The 1Δ put and call bells hold flat past their node** (u − u_i is taken
  as 0 there): beyond 1Δ the smile is the base smile plus that button, so the
  wing carries on at the lifted level instead of dropping back onto the base.
  Continuous with no kink at the node, since a bell is flat at its own centre.

### Final-smile delta vs base-smile delta

`BellConfig(delta=...)`:

- **`"final"` (default).** The bells sit at the true delta of the final smile.
  Because `sigma(K)` appears on both sides, each strike's vol is solved for
  (`strike_vol`):
  1. Every solution lies between `max(0, base - sum|buttons|)` and
     `base + sum|buttons|`, since the bells move vol by at most the sum of the
     buttons. Trial vols across that range, no further apart than 1/256 of
     the top, bracket each solution. More than one bracket raises
     `AmbiguousVolError`.
  2. Newton steps inside the bracket, halving instead whenever a step would
     leave it, until the gap is below 1e-15.

  About 7 ms for the 203 quotes of the one-month SPX slice (was 37 ms with a
  scan from 0 and 64 bisections; same vols to 1e-15 and the same ambiguity
  results on 36 random button sets). Simple repetition (vol → delta → vol)
  does not work: on a one-month SPX slice it already fails to settle with
  ±0.75 vol point alternating buttons.

  `strike_vol_near(..., guess)` runs step 2 alone from a guess (the vols for
  nearby buttons): much faster, but with no ambiguity check. The fitter uses
  it between steps and checks its final answer with `strike_vol`.
  `vol_sensitivity` gives the exact change in each strike's vol per unit of
  each button, including the bells moving the strike's delta
  (see [calibration.md](calibration.md#fast-solver-default)).
- **`"base"`.** The bells sit at the base smile's delta:
  `u = -d1(F, K, tau, base_vol(K))`. Explicit and always unique, but the
  nodes ignore the bells' own effect on delta.

Why it matters: on a steep put wing, delta read off at ATM vol can be far from
the true delta (a 1Δ node at ATM vol can sit on a ~13Δ put).

### Ambiguous vols

In final-delta mode, buttons of alternating sign can give one strike more than
one self-consistent vol. On the one-month test slice (F 7670.65, tau 0.0973),
±1 vol point alternating buttons give three solutions at K = 1.075 F: 0.1160,
0.1204 and 0.1288. `strike_vol` then raises `AmbiguousVolError` (a
`ValueError`) instead of picking one. The scan is best effort: two solutions
closer together than one scan step can slip through, which only happens right
where a second solution first appears.

### Arbitrage

Narrow bells with large buttons can make butterflies negative. Example
(F = 100, 3 months, smooth base smile, 5Δ put +1 vol point): at k = 0.4 the
butterfly around the 5Δ strike goes from 0.018 to -0.041; at k = 1.0 it stays
at +0.008. Accepted for now because the buttons are fitted to market mids.
