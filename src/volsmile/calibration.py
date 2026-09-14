"""Fit a slice in two steps: a least squares spline as the base, then the bell
buttons on top. SVI is fitted as well, only as a comparison.

All fits are least squares on implied vol against market mids, unweighted
(weighting by open interest, volume or spread is left for later).

Bell fit, `fit_bells`: minimise

    sum over quotes (model vol - mid)^2  +  stiffness * sum over buttons (button - start)^2

with every button within +-BUTTON_LIMIT. Stiffness (default 0) is for the
automatic fitter: it keeps the buttons near their previous values, so one
noisy quote moves them less. A button whose bell is below MIN_BELL at every
quote cannot be told from 0 by the quotes, so it is held at its start.

Two solvers for the same problem:

- "fast" (default), `_fit_fast`. Base delta: u is fixed by the base smile, so
  vol = base + bell_matrix(u) @ buttons is linear in the buttons and one bounded
  linear least squares gives the exact minimum. Final delta: Gauss-Newton from
  there (or from `start`). Each step solves every strike's vol at its own
  delta and uses the exact sensitivity of that vol to each button,

      d vol / d button_j = bell_j(u) / (1 - g),   g = (bell slopes . buttons) * d u / d vol,

  (the 1 / (1 - g) is the knock-on of the bells moving the strike's delta),
  then a bounded linear least squares for the step, halved until the objective
  stops rising. It hands over to "scipy" when its maths does not apply: the
  vol floor at 0 binds, a vol is ambiguous, g reaches 1, or it does not
  converge. `BellFit.handover` says why.
- "scipy": `scipy.optimize.least_squares` on the same objective with finite
  differences, first in base-delta mode, then (final delta) refined from there.
  Ambiguous button values are scored as a 1.0 vol error per quote; that is a
  cliff, not a slope, so it can stop early next to the ambiguous region (seen
  with +-2 vol point alternating targets). `BellFit.converged` shows it.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass, fields

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from . import bells, black76, spline, svi
from .marketdata import Slice

BUTTON_LIMIT = 0.2      # 20 vol points
MAX_EVALUATIONS = 300   # per scipy least squares run
FAST_ITERATIONS = 30    # most Gauss-Newton steps (final delta)
FAST_STEP = 1e-7        # converged when no button would move more than this (0.00001 vol points);
                        # much smaller steps change the objective by less than its rounding noise
MIN_BELL = 1e-3         # a bell below this at every quote is held at its start
STALE_ERROR_RATIO = 1.5 # stale spline: fit error above 1.5x the error just after Fit
STALE_BUTTON = 0.02     # stale spline: a fitted button beyond 2 vol points
_AMBIGUOUS = 1.0        # vol error charged per quote when the vols are ambiguous (scipy)
_N = len(fields(bells.BellParams))


@dataclass(frozen=True)
class BellFit:
    buttons: bells.BellParams   # the fitted buttons
    converged: bool             # the solver met its convergence test
    method: str                 # solver that produced the answer: "fast" or "scipy"
    handover: str | None        # why "fast" handed over to "scipy", if it did


@dataclass(frozen=True)
class Fit:
    base: spline.Spline
    buttons: bells.BellParams
    cfg: bells.BellConfig
    rmse_base: float                  # vol, decimals
    rmse_total: float                 # base + bells
    bells_converged: bool             # the bell solver met its convergence test
    svi: svi.SVIParams | None         # comparison only; None if SVI did not converge
    rmse_svi: float | None
    bells_method: str = "fast"        # "fast" or "scipy"
    handover: str | None = None       # why the fast bell fit handed over, if it did


def fit_bells(s: Slice, base_vol, cfg: bells.BellConfig = bells.BellConfig(), method: str = "fast",
              start: bells.BellParams | None = None, stiffness: float = 0.0) -> BellFit:
    """Buttons that bring base + bells closest to the market mid vols (see the
    module docstring). `start`: where the solver starts, and what stiffness
    pulls towards (0 if None)."""
    if method not in ("fast", "scipy"):
        raise ValueError(f'method must be "fast" or "scipy", got {method!r}')
    if not (np.isfinite(stiffness) and stiffness >= 0.0):
        raise ValueError(f"stiffness must be finite and >= 0, got {stiffness!r}")
    q = s.quotes
    K, target = q["strike"].to_numpy(), q["iv"].to_numpy()
    base = np.broadcast_to(np.asarray(base_vol, dtype=float), K.shape)
    prev = np.zeros(_N) if start is None else np.array(astuple(start), dtype=float)
    if prev.shape != (_N,) or not np.all(np.isfinite(prev)):
        raise ValueError("start must be 11 finite buttons")
    prev = np.clip(prev, -BUTTON_LIMIT, BUTTON_LIMIT)
    # Bells that no quote can see are held where they start.
    free = np.max(np.abs(bells.bell_matrix(-black76.d1(s.forward, K, s.tau, base), cfg)), axis=0, initial=0.0) >= MIN_BELL

    handover = None
    if method == "fast":
        x, handover = _fit_fast(K, target, base, s.forward, s.tau, cfg, prev, stiffness, free)
        if x is not None:
            return BellFit(bells.BellParams(*x), True, "fast", None)
    x, converged = _fit_scipy(K, target, base, s.forward, s.tau, cfg, prev, stiffness, free)
    if converged:
        try:  # scipy scores ambiguous buttons with a flat penalty, so it can stop on them
            bells.strike_vol(K, s.forward, s.tau, base, bells.BellParams(*x), cfg)
        except bells.AmbiguousVolError:
            converged = False
    return BellFit(bells.BellParams(*x), converged, "scipy", handover)


def _objective(vol, target, x, prev, stiffness) -> float:
    return float(np.sum((vol - target) ** 2) + stiffness * np.sum((x - prev) ** 2))


def _bounded_lsq(A, b, lower, upper) -> np.ndarray:
    """min |A x - b|^2 with lower <= x <= upper (A has full column rank here)."""
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    if np.all((x >= lower) & (x <= upper)):
        return x
    return lsq_linear(A, b, bounds=(lower, upper), method="bvls").x


def _step(J, residual, x, prev, stiffness, free) -> np.ndarray:
    """Bounded least squares step for the free buttons of
    |J step + residual|^2 + stiffness |x + step - prev|^2 (held buttons do not move)."""
    step = np.zeros(_N)
    if not free.any():
        return step
    root = np.sqrt(stiffness)
    A = np.vstack([J[:, free], root * np.eye(free.sum())])
    b = np.concatenate([-residual, root * (prev - x)[free]])
    step[free] = _bounded_lsq(A, b, (-BUTTON_LIMIT - x)[free], (BUTTON_LIMIT - x)[free])
    return step


def _fit_fast(K, target, base, F, tau, cfg, prev, stiffness, free):
    """(buttons, None) or (None, reason to hand over)."""
    B = bells.bell_matrix(-black76.d1(F, K, tau, base), cfg)
    # Base delta is linear: the exact minimum in one step from prev.
    x = prev + _step(B, base + B @ prev - target, prev, prev, stiffness, free)
    if cfg.delta == "base":
        if np.any(base + B @ x <= 0.0):
            return None, "vol floor at 0 binds"
        return x, None

    # Final delta: Gauss-Newton, starting from the base-delta answer, or from
    # the given start (the automatic fitter's previous buttons).
    if np.any(prev != 0.0):
        x = prev

    def vols(buttons, guess):
        # Between steps: Newton from the last vols, no ambiguity scan (fast).
        return bells.strike_vol_near(K, F, tau, base, bells.BellParams(*buttons), guess, cfg)

    def checked(buttons, vol):
        # The answer: the full solve must agree (unique vol at every strike).
        try:
            full = bells.strike_vol(K, F, tau, base, bells.BellParams(*buttons), cfg)
        except bells.AmbiguousVolError:
            return None, "ambiguous vol"
        if np.max(np.abs(full - vol)) > 1e-12:
            return None, "ambiguous vol"
        return buttons, None

    vol = vols(x, base + B @ x)
    objective = _objective(vol, target, x, prev, stiffness)
    for _ in range(FAST_ITERATIONS):
        if np.any(base + bells.bell_matrix(-black76.d1(F, K, tau, vol), cfg) @ x <= 0.0):
            return None, "vol floor at 0 binds"
        J, g = bells.vol_sensitivity(K, F, tau, vol, bells.BellParams(*x), cfg)
        if np.any(g >= 1.0):
            return None, "vol sensitivity blows up (g >= 1)"
        step = _step(J, vol - target, x, prev, stiffness, free)
        if np.max(np.abs(step)) < FAST_STEP:  # the full Gauss-Newton step is ~0: the minimum
            return checked(x, vol)
        for _ in range(30):  # halve until the objective does not rise
            if np.max(np.abs(step)) < FAST_STEP:  # halved to nothing without improving: stuck, not converged
                return None, "objective would not fall"
            trial = np.clip(x + step, -BUTTON_LIMIT, BUTTON_LIMIT)
            trial_vol = vols(trial, vol)
            trial_objective = _objective(trial_vol, target, trial, prev, stiffness)
            if trial_objective <= objective * (1.0 + 1e-12):  # allow for rounding in the vol solves
                break
            step = step / 2.0
        else:
            return None, "objective would not fall"
        x, vol, objective = trial, trial_vol, trial_objective
    return None, "did not converge"


def _fit_scipy(K, target, base, F, tau, cfg, prev, stiffness, free):
    root = np.sqrt(stiffness)

    def solve(mode, x0):
        c = bells.BellConfig(k=cfg.k, delta=mode)

        def residual(z):
            x = prev.copy()
            x[free] = z
            penalty = root * (x - prev)[free]
            try:
                vol = bells.strike_vol(K, F, tau, base, bells.BellParams(*x), c)
            except bells.AmbiguousVolError:
                return np.concatenate([np.full(target.shape, _AMBIGUOUS), penalty])
            return np.concatenate([vol - target, penalty])

        return least_squares(residual, x0[free], bounds=(-BUTTON_LIMIT, BUTTON_LIMIT),
                             diff_step=1e-6, x_scale=0.01, max_nfev=MAX_EVALUATIONS)

    x = prev.copy()
    if not free.any():
        return x, True
    r = solve("base", prev)
    x[free] = r.x
    if cfg.delta == "final":
        r = solve("final", x)
        x[free] = r.x
    return x, bool(r.status > 0)


def stale_reasons(rmse_now: float, rmse_at_fit: float | None, fitted: bells.BellParams, handover: str | None,
                  error_ratio: float = STALE_ERROR_RATIO, button: float = STALE_BUTTON) -> list[str]:
    """Why the spline base looks stale for the automatic bell fit (empty: fine).
    rmse_now is the bells' fit to the current mids (offsets excluded),
    rmse_at_fit the same just after the last full Fit (None: skip that test)."""
    reasons = []
    if rmse_at_fit is not None and rmse_now > error_ratio * rmse_at_fit:
        reasons.append(f"fit error {100 * rmse_now:.3f} vol pts is above {error_ratio:g}x "
                       f"the {100 * rmse_at_fit:.3f} after Fit")
    big = [f"{name} {100 * v:+.2f}" for name, v in zip(bells.BellParams.__dataclass_fields__, astuple(fitted))
           if abs(v) > button]
    if big:
        reasons.append(f"buttons beyond {100 * button:g} vol pts: {', '.join(big)}")
    if handover == "ambiguous vol":
        reasons.append("the buttons needed make some strike's vol ambiguous")
    return reasons


def rmse(model, target) -> float:
    return float(np.sqrt(np.mean((np.asarray(model) - np.asarray(target)) ** 2)))


def fit_base(s: Slice, knots: int = 3) -> spline.Spline:
    q = s.quotes
    return spline.fit(q["strike"].to_numpy(), s.forward, s.tau, q["iv"].to_numpy(), knots)


def fit_svi(s: Slice) -> svi.SVIParams | None:
    """SVI for comparison; None if it does not converge."""
    try:
        return svi.fit(s.quotes["strike"].to_numpy(), s.forward, s.tau, s.quotes["iv"].to_numpy())
    except RuntimeError:
        return None


def fit_slice(s: Slice, cfg: bells.BellConfig = bells.BellConfig(), knots: int = 3, method: str = "fast") -> Fit:
    K = s.quotes["strike"].to_numpy()
    target = s.quotes["iv"].to_numpy()
    base = fit_base(s, knots)
    base_vol = spline.vol(K, s.forward, base)
    b = fit_bells(s, base_vol, cfg, method)
    total = bells.strike_vol(K, s.forward, s.tau, base_vol, b.buttons, cfg)
    p_svi = fit_svi(s)
    rmse_svi = None if p_svi is None else rmse(svi.vol(K, s.forward, s.tau, p_svi), target)
    return Fit(base, b.buttons, cfg, rmse(base_vol, target), rmse(total, target), b.converged, p_svi, rmse_svi,
               b.method, b.handover)
