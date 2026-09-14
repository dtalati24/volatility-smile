"""The bell model: an overlay of Gaussian bells at delta nodes, added on top of
a base smile (the least squares spline, or anything that gives a vol at each
strike).

    sigma(K) = max(0, base_vol(K) + sum_i button_i * exp(-(u - u_i)^2 / (2 w_i^2)))
    u        = -d1(F, K, tau, sigma(K))
    (for the 1 delta put and call bells, u - u_i is taken as 0 past the node)

u is minus Black-76 d1 at the strike's own final vol, so the put delta is
-N(u) and the call delta N(-u): the bells sit at the true deltas of the final
smile. Nodes are the 1, 2, 5, 10 and 25 delta puts, 50 delta (d1 = 0, just
above the forward) and the same calls, at u_i = N^-1(delta) for puts and
-N^-1(delta) for calls. They bunch up at the tips on purpose, for finer
control there.

Because sigma appears on both sides, each strike's vol is solved for. Buttons
of alternating sign can give a strike more than one self-consistent vol (on a
one-month SPX slice, from about +-1 vol point on neighbouring nodes); that
raises rather than picking one. The check scans 257 vols per strike, so two
solutions closer together than one scan step (just as a second solution
appears) can slip through: best effort, not a proof. BellConfig(delta="base") places the bells at
the base smile's delta instead: u = -d1(F, K, tau, base_vol(K)), explicit and
always unique, at the cost of nodes that ignore the bells' own effect on delta.

Each bell's width is w_i = k * (average distance to its neighbouring nodes), so
tip bells are narrow and middle bells wide. Away from its node every bell
fades out, except that the 1 delta put and call bells hold their full button
past their node: beyond 1 delta the smile is the base smile shifted by that
button, so the wing carries on instead of falling back onto the base.

Narrow bells with large buttons can make butterflies negative (arbitrage);
accepted, since the buttons are fitted to market mids.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass
from statistics import NormalDist

import numpy as np

from .black76 import _inputs, d1

DELTAS = (1, 2, 5, 10, 25)

# Put nodes from the far tip in to 25 delta, 50 delta, then call nodes out to the tip.
_PUT_U = [NormalDist().inv_cdf(d / 100) for d in DELTAS]
NODES = np.array(_PUT_U + [0.0] + [-u for u in reversed(_PUT_U)])

class AmbiguousVolError(ValueError):
    """A strike has more than one self-consistent vol for these buttons."""


_SCAN = 257        # vols tried per strike to bracket the solution
_BISECTIONS = 64   # most Newton-or-halving steps: enough to halve the bracket below one ulp
_TOLERANCE = 1e-15  # stop once every strike's gap (or bracket) is this small


@dataclass(frozen=True)
class BellConfig:
    """k: bell width as a multiple of the distance to the neighbouring nodes.
    delta: "final" places the bells at the final smile's delta, "base" at the
    base smile's."""

    k: float = 0.4
    delta: str = "final"

    def __post_init__(self) -> None:
        if not np.isfinite(self.k) or self.k <= 0.0:
            raise ValueError(f"k must be finite and > 0, got {self.k}")
        if self.delta not in ("final", "base"):
            raise ValueError(f'delta must be "final" or "base", got {self.delta!r}')


@dataclass(frozen=True)
class BellParams:
    """The eleven buttons, in decimals, in the order of NODES. Positive means more vol."""

    put_1: float = 0.0
    put_2: float = 0.0
    put_5: float = 0.0
    put_10: float = 0.0
    put_25: float = 0.0
    atm: float = 0.0
    call_25: float = 0.0
    call_10: float = 0.0
    call_5: float = 0.0
    call_2: float = 0.0
    call_1: float = 0.0


def widths(cfg: BellConfig = BellConfig()) -> np.ndarray:
    """w_i = k * average distance to the neighbouring nodes (one-sided at the tips)."""
    gaps = np.diff(NODES)
    return cfg.k * np.concatenate([[gaps[0]], (gaps[:-1] + gaps[1:]) / 2.0, [gaps[-1]]])


def _distance(u) -> np.ndarray:
    """u minus each node, shape (..., 11). The two 1 delta bells hold their full
    button past their node, out to the far wings: distance 0 there."""
    dist = np.asarray(u, dtype=float)[..., None] - NODES
    dist[..., 0] = np.maximum(dist[..., 0], 0.0)    # below the 1 delta put node
    dist[..., -1] = np.minimum(dist[..., -1], 0.0)  # above the 1 delta call node
    return dist


def bell_matrix(u, cfg: BellConfig = BellConfig()) -> np.ndarray:
    """Each bell's height (for a button of 1) at u, shape (..., 11). The vol the
    bells add is bell_matrix(u) @ buttons, linear in the buttons."""
    dist = _distance(u)
    with np.errstate(over="ignore", invalid="ignore"):  # u = +-inf (vol 0) squares to inf: bell 0
        return np.nan_to_num(np.exp(-(dist ** 2) / (2.0 * widths(cfg) ** 2)), nan=0.0)


def bell_slopes(u, cfg: BellConfig = BellConfig()) -> np.ndarray:
    """d bell_matrix / du, shape (..., 11). Zero past the held 1 delta nodes."""
    w2 = widths(cfg) ** 2
    dist = _distance(u)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.nan_to_num(-dist / w2 * np.exp(-(dist ** 2) / (2.0 * w2)), nan=0.0)


def vol_change(u, p: BellParams, cfg: BellConfig = BellConfig()) -> np.ndarray:
    """Vol the bells add at delta coordinate u = -d1 (decimals)."""
    return bell_matrix(u, cfg) @ np.array(astuple(p))


def _prepare(K, F, tau, base_vol, p: BellParams):
    F, K, tau = _inputs(F, K, tau)
    x = np.array(astuple(p), dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("buttons must be finite")
    base = np.broadcast_to(np.asarray(base_vol, dtype=float), K.shape)
    if not np.all(np.isfinite(base)) or np.any(base <= 0.0):
        raise ValueError("base_vol must be finite and > 0")
    return F, K, tau, base, x


def _newton(F, k, tau, b, x, cfg, vol, lo, hi, safety_net=True):
    """Self-consistent vol inside [lo, hi] (gap >= 0 at lo, <= 0 at hi) from
    `vol`: Newton, halving instead whenever the Newton step would leave the
    bracket or is not at least halving the step before it (so it cannot
    bounce between two points). Any strike still not solved after that is
    bisected on its original bracket."""
    root_tau = np.sqrt(tau)
    lo0, hi0 = lo, hi

    def gap(v):
        d1v = d1(F, k, tau, v)
        raw = b + bell_matrix(-d1v, cfg) @ x
        return np.maximum(0.0, raw) - v, raw, d1v

    last = hi - lo  # size of the step before the last one (Numerical Recipes' rtsafe test)
    moved = hi - lo
    for _ in range(_BISECTIONS):
        f, raw, d1v = gap(vol)
        if np.all((np.abs(f) <= _TOLERANCE) | (hi - lo <= _TOLERANCE)):
            break
        up = f > 0.0
        lo, hi = np.where(up, vol, lo), np.where(up, hi, vol)
        # d gap / d vol = (bell slopes . buttons) * du/dvol - 1, du/dvol = d2 / vol
        slope = np.where(raw > 0.0, (bell_slopes(-d1v, cfg) @ x) * (d1v - vol * root_tau) / vol, 0.0) - 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = f / slope
        step = vol - newton
        use = np.isfinite(step) & (step > lo) & (step < hi) & (np.abs(2.0 * newton) <= last)
        new = np.where(f == 0.0, vol, np.where(use, step, (lo + hi) / 2.0))
        last, moved = moved, np.abs(new - vol)
        vol = new

    # Safety net: plain bisection for anything the loop left unsolved.
    f = gap(vol)[0]
    bad = (np.abs(f) > 1e-12) & safety_net
    if np.any(bad):
        a, z = lo0[bad], hi0[bad]
        kb, bb = k[bad], b[bad]
        for _ in range(200):
            mid = (a + z) / 2.0
            d1m = d1(F, kb, tau, mid)
            up = np.maximum(0.0, bb + bell_matrix(-d1m, cfg) @ x) - mid > 0.0
            a, z = np.where(up, mid, a), np.where(up, z, mid)
            if np.all(z - a <= _TOLERANCE):
                break
        vol = vol.copy()
        vol[bad] = (a + z) / 2.0
    return vol


def _range(b, x):
    """Every solution lies in [low, top]: the bells move vol by at most sum |buttons|,
    so the gap is >= 0 at low = max(0, base - reach) and <= 0 at top = base + reach."""
    reach = np.sum(np.abs(x))
    return np.maximum(0.0, b - reach), b + reach


def strike_vol(K, F: float, tau: float, base_vol, p: BellParams, cfg: BellConfig = BellConfig()) -> np.ndarray:
    """Final vol at strike(s) K: the base smile's vol (at each K, or one number)
    plus the bells, placed at the final vol's own delta. Raises
    AmbiguousVolError if a strike has more than one solution."""
    F, K, tau, base, x = _prepare(K, F, tau, base_vol, p)
    if cfg.delta == "base":
        return np.maximum(0.0, base + bell_matrix(-d1(F, K, tau, base), cfg) @ x)
    k, b = K.ravel(), base.ravel()

    def gap(vol, strikes, bases):
        """Implied minus tried vol; its zero is the answer."""
        u = -d1(F, strikes, tau, np.maximum(vol, 1e-300))
        return np.maximum(0.0, bases + bell_matrix(u, cfg) @ x) - vol

    # Scanning only [low, top], with steps no wider than 1/(_SCAN - 1) of top,
    # finds the same solutions as scanning all of [0, top] and costs far less
    # when the buttons are small.
    low, top = _range(b, x)
    steps = int(np.clip(np.ceil((_SCAN - 1) * np.max((top - low) / top, initial=0.0)), 1, _SCAN - 1))
    tried = low[:, None] + (top - low)[:, None] * np.linspace(0.0, 1.0, steps + 1)
    above = gap(tried, k[:, None], b[:, None]) > 0.0
    falls = above[:, :-1] & ~above[:, 1:]
    # ponytail: two solutions closer than one scan step would be missed; scan finer if that shows up.
    # Best effort, like the scan from 0 it replaced: the grid is never coarser, but it sits at
    # different points, so just as a second solution appears either scan can be the first to see it.
    if np.any(falls.sum(axis=1) + ~above[:, 0] * above.any(axis=1) > 1):
        raise AmbiguousVolError("buttons too large: a strike has more than one self-consistent vol")
    i = np.argmax(falls, axis=1)
    row = np.arange(len(k))
    lo, hi = tried[row, i], tried[row, i + 1]
    vol = _newton(F, k, tau, b, x, cfg, (lo + hi) / 2.0, lo, hi)
    return np.where(above[:, 0], vol, low).reshape(K.shape)  # not above at low: the solution is low itself


def vol_sensitivity(K, F: float, tau: float, vol, p: BellParams, cfg: BellConfig = BellConfig()):
    """How each strike's final vol moves per unit of each button, at the
    self-consistent `vol` for buttons p (final delta). Returns (J, g):

        J[i, j] = bell_j(u_i) / (1 - g_i),   g_i = (bell slopes . buttons) * d u / d vol,

    where d u / d vol = d2 / vol. The 1 / (1 - g) is the knock-on of the bells
    moving the strike's delta. Valid where g < 1 and the vol floor does not
    bind; at g = 1 the strike is at the edge of having two solutions."""
    F, K, tau, _, x = _prepare(K, F, tau, 1.0, p)
    vol = np.broadcast_to(np.asarray(vol, dtype=float), K.shape)
    d1v = d1(F, K, tau, vol)
    g = (bell_slopes(-d1v, cfg) @ x) * (d1v - vol * np.sqrt(tau)) / vol
    return bell_matrix(-d1v, cfg) / (1.0 - g)[..., None], g


def strike_vol_near(K, F: float, tau: float, base_vol, p: BellParams, guess,
                    cfg: BellConfig = BellConfig()) -> np.ndarray:
    """A self-consistent final vol at each strike, found from `guess` (e.g. the
    vols for nearby buttons) without the scan: much faster than strike_vol,
    but with no ambiguity check, so where a strike has several solutions it
    returns one of them. The fitter uses it between steps and checks its final
    answer with strike_vol."""
    F, K, tau, base, x = _prepare(K, F, tau, base_vol, p)
    if cfg.delta == "base":
        return np.maximum(0.0, base + bell_matrix(-d1(F, K, tau, base), cfg) @ x)
    k, b = K.ravel(), base.ravel()
    low, top = _range(b, x)
    start = np.broadcast_to(np.asarray(guess, dtype=float), K.shape).ravel()
    start = np.where(np.isfinite(start), np.clip(start, low, top), (low + top) / 2.0)
    start = np.where(start > 0.0, start, (low + top) / 2.0)  # d1 needs a positive vol
    return _newton(F, k, tau, b, x, cfg, start, low.copy(), top.copy()).reshape(K.shape)
