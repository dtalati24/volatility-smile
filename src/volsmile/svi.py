"""Raw SVI, fitted to the same mids as the spline base, as a comparison.

    k    = ln(K/F)
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))    total variance
    vol  = sqrt(w / tau)

Fitted by least squares to market mid implied vols, unweighted. Butterfly
no-arbitrage is not imposed. The fit keeps b >= 0, |rho| < 1, sigma > 0, the
minimum variance a + b sigma sqrt(1 - rho^2) >= 0, and the wing slopes
b (1 + |rho|) <= 2 (the Lee large-strike bound). Without that last bound a
one-week SPX slice fitted b = 10, rho = 0.99 with the vertex outside the
quotes, which extrapolates to 89% vol a few strikes beyond the last call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def total_variance(k, p: SVIParams) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    return p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.sigma**2))


def vol(K, F: float, tau: float, p: SVIParams) -> np.ndarray:
    """Implied vol at strike(s) K. Raises if the variance goes negative there."""
    w = total_variance(np.log(np.asarray(K, dtype=float) / F), p)
    if np.any(w < 0.0):
        raise ValueError("SVI total variance is negative at some strikes")
    return np.sqrt(w / tau)


def fit(K, F: float, tau: float, market_vol) -> SVIParams:
    """Least squares on vol, several starting points, best one kept."""
    k = np.log(np.asarray(K, dtype=float) / F)
    target = np.asarray(market_vol, dtype=float)
    if k.size < 5 or k.shape != target.shape or not np.all(np.isfinite(target)):
        raise ValueError("need at least 5 finite (strike, vol) pairs of the same shape")
    w_mkt = target**2 * tau
    span = max(np.ptp(k), 1e-3)

    # Fitted: v = minimum variance (instead of a), c = b (1 + |rho|) (instead of b),
    # rho, m, sigma. Box bounds on v and c then give w >= 0 and the wing bound.
    def unpack(x):
        v, c, rho, m, sigma = x
        b = c / (1.0 + abs(rho))
        return SVIParams(v - b * sigma * np.sqrt(1.0 - rho**2), b, rho, m, sigma)

    def residual(x):
        return np.sqrt(np.maximum(total_variance(k, unpack(x)), 0.0) / tau) - target

    lower = [0.0, 0.0, -0.999, k.min() - span, 1e-4]
    upper = [w_mkt.max(), 2.0, 0.999, k.max() + span, 10.0 * span]
    best = None
    for rho in (-0.7, -0.3, 0.0, 0.3):
        for sigma in (0.02 * span, 0.2 * span):
            x0 = [0.5 * w_mkt.min(), 0.5 * (w_mkt.max() - w_mkt.min()) / span, rho, k[np.argmin(target)], sigma]
            x0 = np.clip(x0, lower, upper)
            r = least_squares(residual, x0, bounds=(lower, upper), x_scale="jac", max_nfev=5000)
            if r.status > 0 and (best is None or r.cost < best.cost):
                best = r
    if best is None:
        raise RuntimeError("SVI fit did not converge from any starting point")
    return unpack(best.x)
