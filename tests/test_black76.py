import numpy as np
import pytest
from scipy.stats import norm

from volsmile.black76 import VOL_UPPER, d1, delta, implied_vol, price, vega

F, TAU = 100.0, 0.25
K = np.array([60.0, 80.0, 95.0, 100.0, 105.0, 120.0, 150.0])


def test_known_value_at_the_money():
    # F = K: call = put = F (2 N(vol sqrt(tau) / 2) - 1)
    expected = 100.0 * (2.0 * norm.cdf(0.1) - 1.0)
    assert float(price(100.0, 100.0, 1.0, 0.2, True)) == pytest.approx(expected, rel=1e-14)
    assert expected == pytest.approx(7.965567455405804, rel=1e-12)
    assert float(price(100.0, 100.0, 1.0, 0.2, False)) == pytest.approx(expected, rel=1e-14)


def test_known_value_away_from_the_money():
    s = 0.3 * np.sqrt(0.5)
    a = np.log(100.0 / 90.0) / s + s / 2
    expected = 100.0 * norm.cdf(a) - 90.0 * norm.cdf(a - s)
    assert float(price(100.0, 90.0, 0.5, 0.3, True)) == pytest.approx(expected, rel=1e-14)
    assert float(d1(100.0, 90.0, 0.5, 0.3)) == pytest.approx(a, rel=1e-14)


def test_put_call_parity():
    for vol in (0.05, 0.2, 0.8):
        np.testing.assert_allclose(price(F, K, TAU, vol, True) - price(F, K, TAU, vol, False), F - K, atol=1e-10)


def test_zero_vol_gives_intrinsic_value():
    np.testing.assert_array_equal(price(F, K, TAU, 0.0, True), np.maximum(F - K, 0.0))
    np.testing.assert_array_equal(price(F, K, TAU, 0.0, False), np.maximum(K - F, 0.0))


def test_prices_stay_inside_the_bounds_and_rise_with_vol():
    for c, cap in ((True, F), (False, K)):
        low, high = price(F, K, TAU, 0.1, c), price(F, K, TAU, 0.4, c)
        intrinsic = np.maximum(F - K, 0.0) if c else np.maximum(K - F, 0.0)
        assert np.all(intrinsic <= low) and np.all(low < high) and np.all(high < cap)


def test_delta_is_the_forward_derivative():
    h = 1e-4
    for c in (True, False):
        numeric = (price(F + h, K, TAU, 0.25, c) - price(F - h, K, TAU, 0.25, c)) / (2 * h)
        np.testing.assert_allclose(delta(F, K, TAU, 0.25, c), numeric, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(delta(F, K, TAU, 0.25, True) - delta(F, K, TAU, 0.25, False), 1.0, rtol=1e-14)


def test_vega_is_the_vol_derivative():
    h = 1e-6
    otm = K > F  # out-of-the-money options, so no large intrinsic value cancels
    numeric = (price(F, K, TAU, 0.25 + h, otm) - price(F, K, TAU, 0.25 - h, otm)) / (2 * h)
    np.testing.assert_allclose(vega(F, K, TAU, 0.25), numeric, rtol=1e-6)


def test_25_delta_put_strike_by_hand():
    # put delta -0.25  <=>  d1 = -N^-1(0.25)
    vol, s = 0.2, 0.2 * np.sqrt(TAU)
    strike = F * np.exp(-(-norm.ppf(0.25) - s / 2) * s)
    assert float(delta(F, strike, TAU, vol, False)) == pytest.approx(-0.25, rel=1e-13)


@pytest.mark.parametrize("vol", [0.01, 0.15, 0.6, 2.5])
@pytest.mark.parametrize("is_call", [True, False])
def test_implied_vol_round_trip(vol, is_call):
    # Strikes 0 to 1.5 standard deviations either side, so prices are not
    # just intrinsic value to within rounding.
    strikes = F * np.exp(np.array([-1.5, -0.5, 0.0, 0.5, 1.5]) * vol * np.sqrt(TAU))
    p = price(F, strikes, TAU, vol, is_call)
    got = implied_vol(p, F, strikes, TAU, is_call)
    # Compare in price where vega is tiny, in vol otherwise.
    np.testing.assert_allclose(price(F, strikes, TAU, got, is_call), p, rtol=1e-10, atol=1e-13)
    ok = vega(F, strikes, TAU, vol) > 1e-3
    np.testing.assert_allclose(got[ok], vol, rtol=1e-8)


def test_implied_vol_mixed_calls_and_puts_and_shapes():
    vols = np.array([0.3, 0.25, 0.2, 0.22])
    strikes = np.array([85.0, 95.0, 105.0, 115.0])
    is_call = np.array([False, False, True, True])
    got = implied_vol(price(F, strikes, TAU, vols, is_call), F, strikes, TAU, is_call)
    np.testing.assert_allclose(got, vols, rtol=1e-10)
    assert implied_vol(np.full((2, 3), 5.0), F, 100.0, TAU, True).shape == (2, 3)
    assert np.ndim(implied_vol(5.0, F, 100.0, TAU, True)) == 0


@pytest.mark.parametrize("p, strike, is_call", [
    (-1.0, 100.0, True),          # negative
    (0.0, 110.0, True),           # zero, at intrinsic
    (9.0, 90.0, True),            # below intrinsic 10
    (10.0, 90.0, True),           # exactly intrinsic
    (100.0, 90.0, True),          # at the cap F
    (95.0, 95.0, False),          # at the cap K
    (np.nan, 100.0, True),
])
def test_implied_vol_outside_bounds_is_nan(p, strike, is_call):
    assert np.isnan(implied_vol(p, F, strike, TAU, is_call))


def test_implied_vol_beyond_the_search_range_is_nan():
    too_high = float(price(F, 100.0, TAU, VOL_UPPER * 1.5, True))
    assert np.isnan(implied_vol(too_high, F, 100.0, TAU, True))


@pytest.mark.parametrize("args", [
    (0.0, 100.0, TAU, 0.2), (np.nan, 100.0, TAU, 0.2), (F, -1.0, TAU, 0.2), (F, 0.0, TAU, 0.2),
    (F, 100.0, 0.0, 0.2), (F, 100.0, np.inf, 0.2), (F, 100.0, TAU, -0.1), (F, 100.0, TAU, np.nan),
])
def test_bad_inputs(args):
    with pytest.raises(ValueError):
        price(*args, True)
    with pytest.raises(ValueError):
        delta(*args, True)
    with pytest.raises(ValueError):
        vega(*args)
    F_, K_, tau_, vol_ = args
    if vol_ == 0.2:  # the bad input is F, K or tau, which implied_vol also takes
        with pytest.raises(ValueError):
            implied_vol(5.0, F_, K_, tau_, True)
