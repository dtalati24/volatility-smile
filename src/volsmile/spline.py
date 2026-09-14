"""Least squares cubic spline: the base smile the bells sit on.

Fitted to market mid implied vol against log-moneyness k = ln(K/F):

- Cubic pieces joined at `knots` interior knots, with value, slope and
  curvature matching at each knot. The knots sit at equal quantiles of the
  distinct quoted k (3 knots: the 25%, 50% and 75% points), so they land where
  the quotes are dense. (Distinct, so repeated strikes cannot stack knots on
  top of each other, which would let the slope jump.)
- Least squares: the curve minimises the total squared miss in vol rather
  than passing through every mid.
- Between the lowest and highest quoted k the vol is the spline (floored at 0).
- Beyond them total variance w = vol^2 tau continues in a straight line in k,
  starting from the spline's value and slope at the last quote. The outward
  slope is kept between 0 (the wing never turns down) and 2 (the Lee
  large-strike bound). Inside those limits there is no jump and no kink at
  the last quote; a slope clipped to a limit leaves a kink there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

DEGREE = 3
MAX_WING_SLOPE = 2.0


@dataclass(frozen=True)
class Spline:
    t: tuple[float, ...]   # full knot vector (boundary knots repeated)
    c: tuple[float, ...]   # B-spline coefficients, in vol
    lo: float              # lowest quoted k
    hi: float              # highest quoted k
    tau: float

    @property
    def knots(self) -> int:
        """Number of interior knots."""
        return len(self.t) - 2 * (DEGREE + 1)


def fit(K, F: float, tau: float, market_vol, knots: int = 3) -> Spline:
    """Least squares cubic spline of vol against ln(K/F), `knots` interior knots."""
    if isinstance(knots, bool) or int(knots) != knots or knots < 0:
        raise ValueError(f"knots must be a whole number >= 0, got {knots!r}")
    knots = int(knots)
    F, tau = float(F), float(tau)
    if not (np.isfinite(F) and F > 0.0 and np.isfinite(tau) and tau > 0.0):
        raise ValueError("F and tau must be finite and > 0")
    K = np.asarray(K, dtype=float)
    v = np.asarray(market_vol, dtype=float)
    if K.ndim != 1 or K.shape != v.shape or not np.all(np.isfinite(v)) or np.any(~(K > 0.0)) \
            or not np.all(np.isfinite(K)):
        raise ValueError("strikes and vols must be 1-D, the same length, finite, strikes > 0")
    k = np.log(K / F)
    order = np.argsort(k)
    k, v = k[order], v[order]
    if np.unique(k).size < knots + DEGREE + 1:
        raise ValueError(f"{knots} knots need at least {knots + DEGREE + 1} distinct strikes")
    interior = np.quantile(np.unique(k), np.linspace(0.0, 1.0, knots + 2)[1:-1])
    t = np.r_[[k[0]] * (DEGREE + 1), interior, [k[-1]] * (DEGREE + 1)]
    try:
        s = make_lsq_spline(k, v, t, k=DEGREE)
    except (ValueError, np.linalg.LinAlgError) as e:
        raise ValueError(f"spline fit failed with {knots} knots: {e}") from None
    return Spline(tuple(s.t), tuple(s.c), float(k[0]), float(k[-1]), tau)


def total_variance(k, s: Spline) -> np.ndarray:
    """w = vol^2 tau at log-moneyness k, with straight-line wings beyond the quotes."""
    k = np.asarray(k, dtype=float)
    curve = BSpline(np.array(s.t), np.array(s.c), DEGREE, extrapolate=False)
    slope = curve.derivative()

    def w_and_outward_slope(edge, outward):
        vol = float(curve(edge))
        w = max(vol, 0.0) ** 2 * s.tau
        dw = 2.0 * max(vol, 0.0) * float(slope(edge)) * s.tau * outward
        return w, min(max(dw, 0.0), MAX_WING_SLOPE)

    w_lo, m_lo = w_and_outward_slope(s.lo, -1.0)
    w_hi, m_hi = w_and_outward_slope(s.hi, 1.0)
    inside = np.maximum(curve(np.clip(k, s.lo, s.hi)), 0.0) ** 2 * s.tau
    return np.where(k < s.lo, w_lo + m_lo * (s.lo - k), np.where(k > s.hi, w_hi + m_hi * (k - s.hi), inside))


def vol(K, F: float, s: Spline) -> np.ndarray:
    """Implied vol at strike(s) K."""
    K = np.asarray(K, dtype=float)
    return np.sqrt(total_variance(np.log(K / float(F)), s) / s.tau)
