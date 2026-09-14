from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import BSpline

from volsmile import spline, svi
from volsmile.marketdata import load_slice

F, TAU = 100.0, 0.25
K = F * np.exp(np.linspace(-0.5, 0.2, 60))
DATA = Path(__file__).resolve().parent.parent / "data"


def cubic_vol(K):
    k = np.log(K / F)
    return 0.2 - 0.3 * k + 0.8 * k**2 + 0.5 * k**3


def test_defaults_and_knot_positions():
    s = spline.fit(K, F, TAU, cubic_vol(K))
    assert s.knots == 3
    k = np.log(K / F)
    np.testing.assert_allclose(s.t[4:-4], np.quantile(np.unique(k), [0.25, 0.5, 0.75]), rtol=1e-14)
    assert s.t[:4] == (k[0],) * 4 and s.t[-4:] == (k[-1],) * 4
    assert (s.lo, s.hi, s.tau) == (k[0], k[-1], TAU)


@pytest.mark.parametrize("knots", [0, 1, 3, 6])
def test_any_cubic_is_reproduced_exactly_inside_the_quotes(knots):
    s = spline.fit(K, F, TAU, cubic_vol(K), knots)
    grid = F * np.exp(np.linspace(-0.5, 0.2, 500))
    np.testing.assert_allclose(spline.vol(grid, F, s), cubic_vol(grid), atol=1e-12)


def test_it_is_a_least_squares_fit():
    # Nudging any coefficient can only make the squared vol error larger.
    rng = np.random.default_rng(1)
    target = cubic_vol(K) + 0.004 * rng.standard_normal(K.size)
    s = spline.fit(K, F, TAU, target)
    k = np.log(K / F)

    def sse(c):
        return np.sum((BSpline(np.array(s.t), c, 3)(k) - target) ** 2)

    best = sse(np.array(s.c))
    for i in range(len(s.c)):
        for h in (1e-4, -1e-4):
            c = np.array(s.c)
            c[i] += h
            assert sse(c) > best


def test_unsorted_input_gives_the_same_fit():
    order = np.random.default_rng(2).permutation(K.size)
    a = spline.fit(K, F, TAU, cubic_vol(K))
    b = spline.fit(K[order], F, TAU, cubic_vol(K)[order])
    np.testing.assert_allclose(b.c, a.c, rtol=1e-12)
    assert a.t == b.t


@pytest.mark.parametrize("side", ["put", "call"])
def test_wings_are_straight_lines_in_total_variance_continuing_value_and_slope(side):
    s = spline.fit(K, F, TAU, cubic_vol(K))
    curve = BSpline(np.array(s.t), np.array(s.c), 3)
    edge, outward = (s.lo, -1.0) if side == "put" else (s.hi, 1.0)
    v, dv = float(curve(edge)), float(curve.derivative()(edge))
    w_edge, dw = v**2 * TAU, 2 * v * dv * TAU * outward
    assert 0.0 < dw < 2.0                                     # not clipped for this curve
    d = np.array([0.0, 0.1, 0.5, 2.0])
    k = edge + outward * d
    np.testing.assert_allclose(spline.total_variance(k, s), w_edge + dw * d, rtol=1e-12)
    # No kink: the slope just inside matches the straight line outside.
    h = 1e-6
    inside_slope = (spline.total_variance(edge, s) - spline.total_variance(edge - outward * h, s)) / h
    assert inside_slope == pytest.approx(dw, rel=1e-4)


def test_a_wing_that_turns_down_is_held_flat():
    # Vol falls towards the call edge, so the outward slope is negative: clipped to 0.
    target = 0.2 + 0.5 * np.log(K / F) ** 2 - 1.5 * np.maximum(np.log(K / F) - 0.1, 0) ** 2
    s = spline.fit(K, F, TAU, target)
    k = np.array([s.hi, s.hi + 0.3, s.hi + 3.0])
    w = spline.total_variance(k, s)
    assert w[1] == w[0] and w[2] == w[0]


def test_a_steep_wing_is_capped_at_the_lee_slope():
    target = 0.2 + 30.0 * np.log(K / F) ** 2                  # huge put wing
    s = spline.fit(K, F, TAU, target)
    w = spline.total_variance(np.array([s.lo, s.lo - 1.0, s.lo - 2.0]), s)
    np.testing.assert_allclose(np.diff(w), [2.0, 2.0], rtol=1e-12)


def test_negative_spline_vol_is_floored():
    target = np.where(np.log(K / F) > 0.1, -0.05, 0.2)
    s = spline.fit(K, F, TAU, target, knots=6)
    raw = BSpline(np.array(s.t), np.array(s.c), 3)(np.log(K / F))
    assert np.any(raw < 0.0)                                   # the case really occurs
    np.testing.assert_array_equal(spline.vol(K[raw < 0.0], F, s), 0.0)
    np.testing.assert_allclose(spline.vol(K[raw >= 0.0], F, s), raw[raw >= 0.0], rtol=1e-14)
    if raw[-1] < 0.0:                                         # wing from a zero-vol edge stays at zero
        assert spline.total_variance(s.hi + 1.0, s) == 0.0


def test_repeated_strikes_do_not_stack_knots():
    strikes = np.r_[np.full(20, 100.0), F * np.exp(np.linspace(-0.3, 0.2, 7))]
    s = spline.fit(strikes, F, TAU, cubic_vol(strikes))
    interior = np.array(s.t[4:-4])
    assert np.all(np.diff(interior) > 0)
    np.testing.assert_allclose(interior, np.quantile(np.unique(np.log(strikes / F)), [0.25, 0.5, 0.75]))


def test_scalar_and_array_shapes():
    s = spline.fit(K, F, TAU, cubic_vol(K))
    assert np.ndim(spline.vol(100.0, F, s)) == 0
    assert spline.vol(np.full((3, 2), 90.0), F, s).shape == (3, 2)


@pytest.mark.parametrize("args, kwargs", [
    ((K, F, TAU, cubic_vol(K)), {"knots": -1}),
    ((K, F, TAU, cubic_vol(K)), {"knots": 2.5}),
    ((K, F, TAU, cubic_vol(K)), {"knots": True}),
    ((K[:6], F, TAU, cubic_vol(K[:6])), {"knots": 3}),       # 3 knots need 7 strikes
    ((K, F, TAU, cubic_vol(K)[:-1]), {}),
    ((K, F, TAU, np.r_[cubic_vol(K)[:-1], np.nan]), {}),
    ((np.r_[K[:-1], -1.0], F, TAU, cubic_vol(K)), {}),
    ((K, 0.0, TAU, cubic_vol(K)), {}),
    ((K, F, 0.0, cubic_vol(K)), {}),
])
def test_bad_input_raises(args, kwargs):
    with pytest.raises(ValueError):
        spline.fit(*args, **kwargs)


def test_three_knots_beat_svi_on_the_saved_snapshots():
    for path in sorted(DATA.glob("spx_*.csv")):
        s = load_slice(path)
        q = s.quotes
        sp = spline.fit(q["strike"], s.forward, s.tau, q["iv"])
        base = spline.vol(q["strike"].to_numpy(), s.forward, sp)
        p = svi.fit(q["strike"], s.forward, s.tau, q["iv"])
        rmse_spline = np.sqrt(np.mean((base - q["iv"]) ** 2))
        rmse_svi = np.sqrt(np.mean((svi.vol(q["strike"], s.forward, s.tau, p) - q["iv"]) ** 2))
        assert rmse_spline < rmse_svi
