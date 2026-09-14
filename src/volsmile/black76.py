"""Black-76: European options on a forward, undiscounted (discount factor 1).

    d1 = ln(F/K) / (vol sqrt(tau)) + vol sqrt(tau) / 2,   d2 = d1 - vol sqrt(tau)
    call = F N(d1) - K N(d2),   put = K N(-d2) - F N(-d1)

Undiscounted because the forward comes from put-call parity on the same chain,
which absorbs rates and dividends. Implied vols are computed here from mid
prices against that one forward, rather than taken from Yahoo, whose call and
put vols at the same strike can disagree by several vol points.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

VOL_LOWER, VOL_UPPER = 1e-4, 5.0


def _inputs(F, K, tau, vol=None):
    F, tau = float(F), float(tau)
    for name, value in (("F", F), ("tau", tau)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value}")
    K = np.asarray(K, dtype=float)
    if np.any(~(K > 0.0)) or not np.all(np.isfinite(K)):
        raise ValueError("strikes must be finite and > 0")
    if vol is None:
        return F, K, tau
    vol = np.asarray(vol, dtype=float)
    if np.any(vol < 0.0) or not np.all(np.isfinite(vol)):
        raise ValueError("vols must be finite and >= 0")
    return F, K, tau, vol


def d1(F, K, tau, vol) -> np.ndarray:
    """d1 for vol > 0 (vols of exactly 0 give +-inf, or nan at K = F)."""
    F, K, tau, vol = _inputs(F, K, tau, vol)
    s = vol * np.sqrt(tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(F / K) / s + s / 2.0


def price(F, K, tau, vol, is_call) -> np.ndarray:
    """Undiscounted option price. Vol 0 gives the intrinsic value."""
    F, K, tau, vol = _inputs(F, K, tau, vol)
    s = vol * np.sqrt(tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.log(F / K) / s + s / 2.0
    b = a - s
    call = F * norm.cdf(a) - K * norm.cdf(b)
    put = K * norm.cdf(-b) - F * norm.cdf(-a)
    out = np.where(is_call, call, put)
    intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    return np.where(s > 0.0, out, intrinsic)


def delta(F, K, tau, vol, is_call) -> np.ndarray:
    """Forward delta: N(d1) for calls, N(d1) - 1 for puts. Needs vol > 0."""
    a = d1(F, K, tau, vol)
    return np.where(is_call, norm.cdf(a), -norm.cdf(-a))


def vega(F, K, tau, vol) -> np.ndarray:
    """Price change per 1.00 of vol: F phi(d1) sqrt(tau). Needs vol > 0."""
    return float(F) * norm.pdf(d1(F, K, tau, vol)) * np.sqrt(float(tau))


def implied_vol(prices, F, K, tau, is_call) -> np.ndarray:
    """Vol that reproduces each price, strictly between VOL_LOWER and VOL_UPPER.

    NaN where there is none: the price is at or outside the no-arbitrage
    bounds (intrinsic, F for calls, K for puts), or needs a vol outside the
    search range. Dirty quotes are then dropped instead of stopping a chain.
    """
    F, K, tau = _inputs(F, K, tau)
    prices, K, is_call = np.broadcast_arrays(np.asarray(prices, dtype=float), K, np.asarray(is_call))
    out = np.full(prices.shape, np.nan)
    for i in np.ndindex(prices.shape):
        p, k, c = prices[i], K[i], bool(is_call[i])
        f = lambda v: float(price(F, k, tau, v, c)) - p  # noqa: E731
        # Prices rise with vol, so this also rejects prices at or beyond the
        # bounds, and NaN prices (every comparison with NaN is False).
        if f(VOL_LOWER) < 0.0 < f(VOL_UPPER):
            out[i] = brentq(f, VOL_LOWER, VOL_UPPER, xtol=1e-14, rtol=1e-14, maxiter=200)
    return out
