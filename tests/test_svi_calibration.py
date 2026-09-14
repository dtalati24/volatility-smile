from dataclasses import astuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volsmile import bells, spline, svi
from volsmile.calibration import BUTTON_LIMIT, fit_bells, fit_slice
from volsmile.marketdata import Slice, load_slice

F, TAU = 7600.0, 0.1
TRUE_SVI = svi.SVIParams(a=0.0008, b=0.03, rho=-0.7, m=0.01, sigma=0.04)
DATA = Path(__file__).resolve().parent.parent / "data"


def make_slice(K, vols):
    q = pd.DataFrame({"strike": K, "iv": vols})
    return Slice("2026-10-16", pd.Timestamp("2026-09-11 20:00", tz="UTC"), F, TAU, q)


def test_total_variance_formula():
    k = np.array([-0.2, 0.0, 0.1])
    p = TRUE_SVI
    expected = p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.sigma**2))
    np.testing.assert_allclose(svi.total_variance(k, p), expected, rtol=1e-15)
    np.testing.assert_allclose(svi.vol(F * np.exp(k), F, TAU, p), np.sqrt(expected / TAU), rtol=1e-14)


def test_negative_variance_raises():
    with pytest.raises(ValueError):
        svi.vol([F], F, TAU, svi.SVIParams(a=-0.01, b=0.01, rho=0.0, m=0.0, sigma=0.1))


@pytest.mark.parametrize("p", [
    TRUE_SVI,
    svi.SVIParams(a=0.004, b=0.08, rho=-0.3, m=-0.05, sigma=0.15),
    svi.SVIParams(a=0.0001, b=0.02, rho=0.2, m=0.03, sigma=0.02),
])
def test_svi_fit_recovers_an_exact_svi_smile(p):
    K = F * np.exp(np.linspace(-0.35, 0.12, 80))
    target = svi.vol(K, F, TAU, p)
    got = svi.fit(K, F, TAU, target)
    np.testing.assert_allclose(svi.vol(K, F, TAU, got), target, atol=1e-7)
    assert got.b >= 0 and abs(got.rho) < 1 and got.sigma > 0
    assert got.a + got.b * got.sigma * np.sqrt(1 - got.rho**2) >= -1e-15
    assert got.b * (1 + abs(got.rho)) <= 2 + 1e-12


def test_svi_fit_respects_the_wing_bound_on_the_one_week_slice():
    s = load_slice(DATA / "spx_2026-09-21_2026-09-11.csv")
    got = svi.fit(s.quotes["strike"], s.forward, s.tau, s.quotes["iv"])
    assert got.b * (1 + abs(got.rho)) <= 2 + 1e-12
    fitted = svi.vol(s.quotes["strike"], s.forward, s.tau, got)
    assert np.sqrt(np.mean((fitted - s.quotes["iv"]) ** 2)) < 0.004


@pytest.mark.parametrize("args", [
    ([1.0, 2.0, 3.0], F, TAU, [0.2, 0.2, 0.2]),                    # too few
    (np.linspace(7000, 8000, 10), F, TAU, np.full(9, 0.2)),        # shapes differ
    (np.linspace(7000, 8000, 10), F, TAU, np.r_[np.full(9, 0.2), np.nan]),
])
def test_svi_fit_rejects_bad_input(args):
    with pytest.raises(ValueError):
        svi.fit(*args)


@pytest.mark.parametrize("delta", ["final", "base"])
def test_bell_fit_recovers_known_buttons(delta):
    cfg = bells.BellConfig(delta=delta)
    true = bells.BellParams(put_1=0.004, put_2=-0.002, put_5=0.003, put_10=0.001, put_25=-0.002, atm=0.001,
                            call_25=0.002, call_10=-0.001, call_5=0.002, call_2=0.001, call_1=-0.002)
    K = F * np.exp(np.linspace(-0.45, 0.15, 400))
    base = svi.vol(K, F, TAU, TRUE_SVI)
    s = make_slice(K, bells.strike_vol(K, F, TAU, base, true, cfg))
    for method in ("fast", "scipy"):
        got = fit_bells(s, base, cfg, method)
        assert got.converged and got.method == method and got.handover is None
        np.testing.assert_allclose(astuple(got.buttons), astuple(true), atol=1e-7 if method == "fast" else 2e-5)


def test_buttons_with_no_quotes_near_their_node_stay_at_zero():
    K = F * np.exp(np.linspace(-0.03, 0.03, 60))          # near the money only
    base = svi.vol(K, F, TAU, TRUE_SVI)
    s = make_slice(K, base + 0.003)
    for method in ("fast", "scipy"):
        got = fit_bells(s, base, method=method).buttons
        assert got.put_1 == 0.0 and got.call_1 == 0.0
        assert all(abs(v) <= BUTTON_LIMIT for v in astuple(got))


def test_fit_slice_on_a_saved_spx_snapshot():
    s = load_slice(DATA / "spx_2026-10-16_2026-09-11.csv")
    fit = fit_slice(s)
    assert fit.base.knots == 3 and fit.bells_converged and fit.bells_method == "fast" and fit.handover is None
    assert fit.rmse_base < fit.rmse_svi < 0.005        # spline beats SVI; both within half a vol point
    assert fit.rmse_total < 0.8 * fit.rmse_base        # the bells take out a real part of what is left
    K = s.quotes["strike"].to_numpy()
    target = s.quotes["iv"].to_numpy()
    base = spline.vol(K, s.forward, fit.base)
    total = bells.strike_vol(K, s.forward, s.tau, base, fit.buttons, fit.cfg)
    assert np.sqrt(np.mean((base - target) ** 2)) == pytest.approx(fit.rmse_base, rel=1e-12)
    assert np.sqrt(np.mean((total - target) ** 2)) == pytest.approx(fit.rmse_total, rel=1e-12)
    comparison = svi.vol(K, s.forward, s.tau, fit.svi)
    assert np.sqrt(np.mean((comparison - target) ** 2)) == pytest.approx(fit.rmse_svi, rel=1e-12)


def test_knots_setting_is_used():
    s = load_slice(DATA / "spx_2026-10-16_2026-09-11.csv")
    stiff, flexible = fit_slice(s, knots=1), fit_slice(s, knots=6)
    assert (stiff.base.knots, flexible.base.knots) == (1, 6)
    assert flexible.rmse_base < stiff.rmse_base
