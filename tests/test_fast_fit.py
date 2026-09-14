"""The fast bell fitter: same answer as scipy, exact sensitivities, handovers,
stiffness, held buttons, and the stale-spline check."""
from dataclasses import astuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volsmile import bells, calibration, spline
from volsmile.bells import BellConfig, BellParams
from volsmile.calibration import BUTTON_LIMIT, fit_base, fit_bells, rmse, stale_reasons
from volsmile.marketdata import Slice, load_slice

DATA = Path(__file__).resolve().parent.parent / "data"
SLICES = sorted(p.name for p in DATA.glob("*_*.csv"))
NAMES = list(BellParams.__dataclass_fields__)
F, TAU, ATM = 7670.65, 0.0973, 0.1405


def smooth_base(K):
    z = np.log(K / F) / (ATM * np.sqrt(TAU))
    return ATM - 0.03 * z + 0.012 * z**2


def synthetic(K, vols):
    return Slice("2026-10-16", pd.Timestamp("2026-09-11 20:00", tz="UTC"), F, TAU,
                 pd.DataFrame({"strike": K, "iv": vols}))


def model_vol(s, base, buttons, cfg):
    return bells.strike_vol(s.quotes["strike"].to_numpy(), s.forward, s.tau, base, buttons, cfg)


# ---------------------------------------------------------------- building blocks

def test_bell_matrix_is_vol_change_per_button():
    u = np.linspace(-4, 4, 161)
    p = BellParams(*np.linspace(-0.01, 0.01, 11))
    np.testing.assert_allclose(bells.bell_matrix(u) @ np.array(astuple(p)), bells.vol_change(u, p), rtol=0, atol=1e-18)
    assert bells.bell_matrix(u).shape == (161, 11)


@pytest.mark.parametrize("cfg", [BellConfig(), BellConfig(k=0.8)])
def test_bell_slopes_match_finite_differences_including_the_held_tips(cfg):
    u = np.r_[np.linspace(-3.5, 3.5, 141), bells.NODES + 0.013, bells.NODES - 0.013]
    h = 1e-6
    fd = (bells.bell_matrix(u + h, cfg) - bells.bell_matrix(u - h, cfg)) / (2 * h)
    np.testing.assert_allclose(bells.bell_slopes(u, cfg), fd, rtol=1e-6, atol=1e-8)
    beyond = np.array([-3.0, 3.0])
    assert bells.bell_slopes(beyond, cfg)[0, 0] == 0.0 and bells.bell_slopes(beyond, cfg)[1, -1] == 0.0


def test_vol_sensitivity_matches_nudged_full_solves():
    K = np.linspace(0.8, 1.1, 41) * F
    base = smooth_base(K)
    p = BellParams(put_1=0.006, put_2=0.004, put_5=0.0025, put_10=0.0015, put_25=0.0005, atm=-0.003,
                   call_25=-0.004, call_10=0.002, call_5=0.01, call_2=0.002, call_1=0.003)
    vol = bells.strike_vol(K, F, TAU, base, p)
    J, g = bells.vol_sensitivity(K, F, TAU, vol, p)
    assert np.max(np.abs(g)) > 0.05                     # the knock-on term is really exercised
    h = 1e-7
    x = np.array(astuple(p))
    for j in range(11):
        up, down = x.copy(), x.copy()
        up[j] += h
        down[j] -= h
        fd = (bells.strike_vol(K, F, TAU, base, BellParams(*up)) - bells.strike_vol(K, F, TAU, base, BellParams(*down))) / (2 * h)
        np.testing.assert_allclose(J[:, j], fd, rtol=1e-5, atol=1e-8)
    # Without the 1 / (1 - g) knock-on the sensitivity would be visibly wrong.
    plain = bells.bell_matrix(-bells.d1(F, K, TAU, vol))
    assert np.max(np.abs(plain - J)) > 1e-3


def test_strike_vol_near_finds_the_same_vols_from_a_rough_guess():
    K = np.linspace(0.8, 1.1, 121) * F
    base = smooth_base(K)
    rng = np.random.default_rng(7)
    for _ in range(10):
        p = BellParams(*rng.normal(0.0, 0.004, 11))
        full = bells.strike_vol(K, F, TAU, base, p)
        for guess in (base, full + 0.01, np.full(K.shape, 5.0), np.full(K.shape, np.nan)):
            np.testing.assert_allclose(bells.strike_vol_near(K, F, TAU, base, p, guess), full, rtol=0, atol=2e-15)


def test_strike_vol_near_returns_a_solution_where_strike_vol_refuses():
    alternating = BellParams(**{n: 0.01 * (-1) ** i for i, n in enumerate(NAMES)})
    strike = np.array([1.075 * F])
    base = smooth_base(strike)
    with pytest.raises(bells.AmbiguousVolError):
        bells.strike_vol(strike, F, TAU, base, alternating)
    vol = bells.strike_vol_near(strike, F, TAU, base, alternating, base)
    u = -bells.d1(F, strike, TAU, vol)
    assert abs(float(base[0] + bells.vol_change(u, alternating)[0] - vol[0])) < 1e-14


# ---------------------------------------------------------------- same answer as scipy

@pytest.mark.parametrize("delta", ["base", "final"])
@pytest.mark.parametrize("name", SLICES)
def test_fast_matches_scipy_on_the_saved_slices(name, delta):
    s = load_slice(DATA / name)
    K = s.quotes["strike"].to_numpy()
    base = spline.vol(K, s.forward, fit_base(s))
    cfg = BellConfig(delta=delta)
    fast, slow = fit_bells(s, base, cfg), fit_bells(s, base, cfg, "scipy")
    assert (fast.method, fast.handover, fast.converged) == ("fast", None, True)
    assert slow.converged
    target = s.quotes["iv"].to_numpy()
    e_fast, e_slow = rmse(model_vol(s, base, fast.buttons, cfg), target), rmse(model_vol(s, base, slow.buttons, cfg), target)
    assert e_fast <= e_slow + 1e-9                      # never worse, to 1e-7 vol points
    np.testing.assert_allclose(astuple(fast.buttons), astuple(slow.buttons), atol=1e-5)  # 0.001 vol points


@pytest.mark.parametrize("delta", ["base", "final"])
def test_warm_start_lands_on_the_same_minimum(delta):
    s = load_slice(DATA / "spx_2026-10-16_2026-09-11.csv")
    K = s.quotes["strike"].to_numpy()
    base = spline.vol(K, s.forward, fit_base(s))
    cfg = BellConfig(delta=delta)
    cold = fit_bells(s, base, cfg)
    nearby = BellParams(*(np.array(astuple(cold.buttons)) + 0.002 * (-1) ** np.arange(11)))
    warm = fit_bells(s, base, cfg, start=nearby)
    assert warm.method == "fast"
    np.testing.assert_allclose(astuple(warm.buttons), astuple(cold.buttons), atol=1e-6)


def test_warm_start_from_the_answer_needs_less_work(monkeypatch):
    s = load_slice(DATA / "spx_2026-09-21_2026-09-11.csv")
    K = s.quotes["strike"].to_numpy()
    base = spline.vol(K, s.forward, fit_base(s))
    calls = {"n": 0}
    real = bells.strike_vol_near

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)
    monkeypatch.setattr(bells, "strike_vol_near", counting)
    cold = fit_bells(s, base)
    cold_calls, calls["n"] = calls["n"], 0
    warm = fit_bells(s, base, start=cold.buttons)
    assert warm.method == "fast" and calls["n"] <= 2 < cold_calls
    np.testing.assert_allclose(astuple(warm.buttons), astuple(cold.buttons), atol=1e-6)


# ---------------------------------------------------------------- stiffness and held buttons

def test_stiffness_trades_fit_error_for_staying_near_the_start():
    s = load_slice(DATA / "spx_2026-10-16_2026-09-11.csv")
    K, target = s.quotes["strike"].to_numpy(), s.quotes["iv"].to_numpy()
    base = spline.vol(K, s.forward, fit_base(s))
    start = BellParams()
    errors, moves = [], []
    for stiffness in (0.0, 1e-4, 1e-2, 1.0, 1e4):
        got = fit_bells(s, base, start=start, stiffness=stiffness)
        assert got.method == "fast"
        errors.append(rmse(model_vol(s, base, got.buttons, BellConfig()), target))
        moves.append(np.sum(np.array(astuple(got.buttons)) ** 2))
    assert all(a <= b + 1e-12 for a, b in zip(errors, errors[1:]))    # stiffer: fit error never falls
    assert all(a >= b - 1e-15 for a, b in zip(moves, moves[1:]))      # stiffer: buttons never move further
    assert moves[-1] < 1e-4 * moves[0]                                # very stiff: they barely move
    slow = fit_bells(s, base, start=start, stiffness=1e-2, method="scipy")
    np.testing.assert_allclose(astuple(fit_bells(s, base, start=start, stiffness=1e-2).buttons),
                               astuple(slow.buttons), atol=1e-5)


def test_faintly_visible_buttons_are_held_instead_of_fitting_noise(monkeypatch):
    # Quotes down to 14% below the forward: the 2 delta put bell is at most 4e-7 at any quote.
    K = F * np.exp(np.linspace(-0.15, 0.03, 60))
    base = smooth_base(K)
    s = synthetic(K, base + np.random.default_rng(0).normal(0.0, 3e-4, K.shape))
    B = bells.bell_matrix(-bells.d1(F, K, TAU, base))
    assert 0.0 < B[:, 1].max() < calibration.MIN_BELL
    assert fit_bells(s, base).buttons.put_2 == 0.0
    monkeypatch.setattr(calibration, "MIN_BELL", 0.0)              # without holding it...
    assert abs(fit_bells(s, base).buttons.put_2) > 0.05              # ...it swings by 5+ vol points on noise


def test_buttons_no_quote_can_see_stay_at_their_start():
    K = F * np.exp(np.linspace(-0.03, 0.03, 60))      # near the money only
    base = smooth_base(K)
    s = synthetic(K, base + 0.003)
    start = BellParams(put_1=0.004, call_1=-0.003)
    for method in ("fast", "scipy"):
        got = fit_bells(s, base, method=method, start=start).buttons
        assert (got.put_1, got.call_1) == (0.004, -0.003)


# ---------------------------------------------------------------- handovers

def test_vol_floor_hands_over_to_scipy():
    # Base 2 vol points, mids 0.01: the linear least squares would push some vols below 0.
    K = np.linspace(0.97, 1.12, 80) * F
    base = np.full(K.shape, 0.02)
    s = synthetic(K, np.full(K.shape, 1e-4))
    cfg = BellConfig(delta="base")
    B = bells.bell_matrix(-bells.d1(F, K, TAU, base), cfg)
    assert np.min(base + B @ np.linalg.lstsq(B, 1e-4 - base, rcond=None)[0]) < 0.0   # the case really occurs
    got = fit_bells(s, base, cfg)
    assert (got.method, got.handover, got.converged) == ("scipy", "vol floor at 0 binds", True)
    # Final delta has no such kink here (a vol near 0 puts the strike far from every node):
    # the fast fit stays fast and matches the mids at least as well as scipy.
    fast, slow = fit_bells(s, base), fit_bells(s, base, method="scipy")
    assert fast.method == "fast"
    target = np.full(K.shape, 1e-4)
    assert rmse(model_vol(s, base, fast.buttons, BellConfig()), target) <= rmse(model_vol(s, base, slow.buttons, BellConfig()), target) + 1e-12


@pytest.mark.parametrize("patch, reason", [
    ("ambiguous", "ambiguous vol"),
    ("g", "vol sensitivity blows up (g >= 1)"),
    ("iterations", "did not converge"),
    ("objective", "objective would not fall"),
])
def test_every_handover_reason_reaches_scipy(monkeypatch, patch, reason):
    s = load_slice(DATA / "spx_2026-10-16_2026-09-11.csv")
    K = s.quotes["strike"].to_numpy()
    base = spline.vol(K, s.forward, fit_base(s))
    if patch == "ambiguous":
        real = bells.strike_vol
        calls = {"n": 0}

        def refuse_first(*args, **kwargs):   # the fast fitter's final check sees an ambiguous vol
            calls["n"] += 1
            if calls["n"] == 1:
                raise bells.AmbiguousVolError("test")
            return real(*args, **kwargs)
        monkeypatch.setattr(bells, "strike_vol", refuse_first)
    elif patch == "g":
        real = bells.vol_sensitivity
        monkeypatch.setattr(bells, "vol_sensitivity", lambda *a, **k: (real(*a, **k)[0], np.ones(len(K))))
    elif patch == "iterations":
        monkeypatch.setattr(calibration, "FAST_ITERATIONS", 1)
    else:
        real = calibration._objective
        count = {"n": 0}

        def rising(*args):
            count["n"] += 1
            return real(*args) + count["n"]
        monkeypatch.setattr(calibration, "_objective", rising)
    got = fit_bells(s, base)
    assert (got.method, got.handover) == ("scipy", reason)


def test_a_scipy_answer_with_ambiguous_vols_is_not_called_converged(monkeypatch):
    s = load_slice(DATA / "spx_2026-10-16_2026-09-11.csv")
    base = spline.vol(s.quotes["strike"].to_numpy(), s.forward, fit_base(s))
    alternating = np.array([0.05 * (-1) ** i for i in range(11)])
    monkeypatch.setattr(calibration, "_fit_scipy", lambda *args: (alternating, True))
    got = fit_bells(s, base, method="scipy")
    assert got.method == "scipy" and not got.converged


def test_bad_arguments():
    s = load_slice(DATA / "spx_2026-10-16_2026-09-11.csv")
    base = spline.vol(s.quotes["strike"].to_numpy(), s.forward, fit_base(s))
    for kwargs in ({"method": "newton"}, {"stiffness": -1.0}, {"stiffness": np.nan},
                   {"start": BellParams(put_1=np.inf)}):
        with pytest.raises(ValueError):
            fit_bells(s, base, **kwargs)


def test_start_outside_the_limits_is_clipped():
    s = load_slice(DATA / "spx_2026-10-16_2026-09-11.csv")
    base = spline.vol(s.quotes["strike"].to_numpy(), s.forward, fit_base(s))
    got = fit_bells(s, base, start=BellParams(atm=0.5), stiffness=1e6)
    assert 0.199 < got.buttons.atm <= BUTTON_LIMIT      # pulled towards the clipped start 0.2, not 0.5


# ---------------------------------------------------------------- automatic fitter scenario and stale check

def test_stale_reasons():
    ok = BellParams(atm=0.01)
    assert stale_reasons(0.0012, 0.001, ok, None) == []
    assert "above 1.5x" in stale_reasons(0.0016, 0.001, ok, None)[0]
    assert stale_reasons(0.0016, 0.001, ok, None, error_ratio=2.0) == []
    big = stale_reasons(0.001, 0.001, BellParams(put_10=-0.025, atm=0.021), None)
    assert big == ["buttons beyond 2 vol pts: put_10 -2.50, atm +2.10"]
    assert stale_reasons(0.001, 0.001, ok, "ambiguous vol") == ["the buttons needed make some strike's vol ambiguous"]
    assert stale_reasons(0.001, 0.001, ok, "did not converge") == []
    assert stale_reasons(0.5, None, ok, None) == []                     # no post-Fit error: ratio test skipped


def test_auto_fit_with_a_fixed_spline_flags_a_market_move_but_not_noise():
    s = load_slice(DATA / "spx_2026-10-16_2026-09-11.csv")
    K, mids = s.quotes["strike"].to_numpy(), s.quotes["iv"].to_numpy()
    base = spline.vol(K, s.forward, fit_base(s))
    first = fit_bells(s, base)
    at_fit = rmse(model_vol(s, base, first.buttons, BellConfig()), mids)

    def auto(new_mids):
        moved = Slice(s.expiry, s.as_of, s.forward, s.tau, s.quotes.assign(iv=new_mids))
        got = fit_bells(moved, base, start=first.buttons)          # spline kept fixed
        now = rmse(model_vol(moved, base, got.buttons, BellConfig()), new_mids)
        return got, stale_reasons(now, at_fit, got.buttons, got.handover)

    noisy, reasons = auto(mids + np.random.default_rng(3).normal(0.0, 0.0002, K.shape))
    assert noisy.method == "fast" and reasons == []
    shifted, reasons = auto(mids + 0.03)                              # whole smile +3 vol points
    assert reasons                                                    # the bells cannot make a flat shift


def test_newton_does_not_stall_between_two_points():
    # Found in review: the old loop bounced between 0.3159 and 0.3966 and returned 0.2731, not a root.
    K, tau, base = 3940.8593675634056, 1.8178945385126455, 0.27346891803478146
    p = BellParams(-0.0002, -0.0347, -0.00525, 0.00572, 0.01379, 0.01995, 0.00507, 0.03606, -0.02748, -0.00656, -0.03027)
    full = bells.strike_vol([K], F, tau, base, p)
    np.testing.assert_allclose(bells.strike_vol_near([K], F, tau, base, p, [base]), full, rtol=0, atol=1e-14)
    assert float(full[0]) == pytest.approx(0.24782033, abs=1e-8)
    # Newton itself (without the bisection safety net) must get there too.
    x = np.array(astuple(p))
    low, top = bells._range(np.array([base]), x)
    vol = bells._newton(F, np.array([K]), tau, np.array([base]), x, BellConfig(), np.array([base]), low, top,
                        safety_net=False)
    np.testing.assert_allclose(vol, full, rtol=0, atol=1e-14)


def test_every_returned_vol_is_self_consistent_on_hard_random_cases():
    rng = np.random.default_rng(11)
    for _ in range(300):
        tau = rng.uniform(0.01, 2.0)
        K = F * np.exp(rng.uniform(-0.8, 0.4, 30))
        base = rng.uniform(0.08, 0.5, 30)
        p = BellParams(*rng.normal(0.0, rng.choice([0.005, 0.02, 0.04]), 11))
        results = [bells.strike_vol_near(K, F, tau, base, p, guess) for guess in (base, base * rng.uniform(0.2, 3.0, 30))]
        try:
            results.append(bells.strike_vol(K, F, tau, base, p))
        except bells.AmbiguousVolError:
            pass
        for vol in results:
            u = -bells.d1(F, K, tau, np.maximum(vol, 1e-300))
            assert np.max(np.abs(np.maximum(0.0, base + bells.vol_change(u, p)) - vol)) < 1e-13


# ---------------------------------------------------------------- the no-scan shortcut

def test_certified_strikes_really_have_one_solution():
    # Wherever the bound says "one solution", a dense scan shows the gap falling all the way.
    rng = np.random.default_rng(5)
    certified = 0
    for _ in range(120):
        tau = rng.uniform(0.01, 2.0)
        K = F * np.exp(rng.uniform(-0.8, 0.4, 20))
        base = rng.uniform(0.08, 0.5, 20)
        p = BellParams(*rng.normal(0.0, rng.choice([0.002, 0.005, 0.02]), 11))
        x = np.array(astuple(p))
        low, top = bells._range(base, x)
        for i in np.flatnonzero(bells._one_solution(F, K, tau, x, BellConfig(), low, top)):
            vol = np.linspace(low[i], top[i], 2001)
            u = -bells.d1(F, np.full(vol.shape, K[i]), tau, vol)
            gap = np.maximum(0.0, base[i] + bells.vol_change(u, p)) - vol
            assert np.all(np.diff(gap) < 0.0)
            certified += 1
    assert certified > 500                                            # the shortcut is really exercised


def test_shortcut_gives_the_same_vols_and_ambiguity_as_scanning_every_strike(monkeypatch):
    rng = np.random.default_rng(9)
    cases = []
    for _ in range(300):
        tau = rng.uniform(0.01, 2.0)
        K = F * np.exp(rng.uniform(-0.8, 0.4, 30))
        base = rng.uniform(0.08, 0.5, 30)
        cases.append((K, tau, base, BellParams(*rng.normal(0.0, rng.choice([0.001, 0.003, 0.008, 0.02]), 11))))

    def solve_all():
        out = []
        for K, tau, base, p in cases:
            try:
                out.append(bells.strike_vol(K, F, tau, base, p))
            except bells.AmbiguousVolError:
                out.append(None)
        return out

    fast = solve_all()
    monkeypatch.setattr(bells, "_one_solution", lambda F, k, *args: np.zeros(k.shape, dtype=bool))
    scanned = solve_all()
    assert sum(v is None for v in scanned) > 10 and sum(v is not None for v in scanned) > 100
    for a, b in zip(fast, scanned):
        assert (a is None) == (b is None)
        if a is not None:
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-14)


@pytest.mark.parametrize("name", SLICES)
def test_most_real_strikes_skip_the_scan(name):
    s = load_slice(DATA / name)
    K = s.quotes["strike"].to_numpy()
    base = spline.vol(K, s.forward, fit_base(s))
    x = np.array(astuple(fit_bells(s, base).buttons))
    low, top = bells._range(base, x)
    assert bells._one_solution(s.forward, K, s.tau, x, BellConfig(), low, top).mean() >= 0.85


def test_heights_and_slopes_from_one_exponential_match_including_infinite_u():
    u = np.r_[np.linspace(-4, 4, 81), -np.inf, np.inf]
    for cfg in (BellConfig(), BellConfig(k=0.7)):
        heights, slopes = bells._bells_and_slopes(u, cfg)
        np.testing.assert_array_equal(heights, bells.bell_matrix(u, cfg))
        np.testing.assert_array_equal(slopes, bells.bell_slopes(u, cfg))
        assert np.all(np.isfinite(slopes)) and heights[-2, 0] == 1.0 and heights[-1, -1] == 1.0
        assert heights[-2, 1:].sum() == 0.0 and heights[-1, :-1].sum() == 0.0


def test_widths_are_cached_and_read_only():
    w = bells.widths(BellConfig(k=0.4))
    assert w is bells.widths(BellConfig(k=0.4))
    with pytest.raises(ValueError):
        w[0] = 1.0


def test_vol_sensitivity_rejects_a_zero_vol():
    with pytest.raises(ValueError):
        bells.vol_sensitivity([F, F], F, TAU, [0.2, 0.0], BellParams(atm=0.01))


def test_u_span_covers_every_vol_in_the_range_including_the_turning_point():
    rng = np.random.default_rng(13)
    turned = 0
    for _ in range(300):
        tau = rng.uniform(0.01, 2.0)
        K = F * np.exp(rng.uniform(-0.8, 0.8, 1))
        low = rng.uniform(0.01, 0.5, 1)
        top = low + rng.uniform(0.001, 0.8, 1)
        umin, umax = bells._u_span(F, K, tau, low, top)
        vol = np.linspace(low[0], top[0], 4001)
        u = -bells.d1(F, np.full(vol.shape, K[0]), tau, vol)
        assert umin[0] <= u.min() + 1e-12 and umax[0] >= u.max() - 1e-12
        turned += bool(np.log(F / K[0]) > 0 and low[0] < np.sqrt(2 * np.log(F / K[0]) / tau) < top[0])
    assert turned > 20                                                # the turning-point case is exercised


def test_max_abs_slope_bounds_the_slope_over_any_interval():
    rng = np.random.default_rng(17)
    cfg = BellConfig()
    w = bells.widths(cfg)
    starts = np.r_[rng.uniform(-3.5, 3.5, 200), bells.NODES - w - 0.01, bells.NODES + w - 0.01]   # incl. just below each peak
    lengths = np.r_[rng.uniform(0.0, 1.0, 200), np.full(22, 0.02)]
    umin, umax = starts, starts + lengths
    bound = bells._max_abs_slope(umin, umax, cfg)
    for i in range(len(umin)):
        u = np.linspace(umin[i], umax[i], 2001)
        exact = np.abs(bells.bell_slopes(u, cfg)).max(axis=0)
        assert np.all(bound[i] >= exact - 1e-12)
    # tight at a peak: the interval around node - w reaches the peak value
    j = 5
    around = bells._max_abs_slope(np.array([bells.NODES[j] - w[j] - 0.01]), np.array([bells.NODES[j] - w[j] + 0.01]), cfg)
    assert around[0, j] == pytest.approx(1.0 / (w[j] * np.sqrt(np.e)), rel=1e-12)


def test_a_zero_vol_between_steps_hands_over(monkeypatch):
    s = load_slice(DATA / "spx_2026-10-16_2026-09-11.csv")
    base = spline.vol(s.quotes["strike"].to_numpy(), s.forward, fit_base(s))
    real = bells.strike_vol_near

    def with_a_zero(*args, **kwargs):
        vol = real(*args, **kwargs).copy()
        vol[0] = 0.0
        return vol
    monkeypatch.setattr(bells, "strike_vol_near", with_a_zero)
    got = fit_bells(s, base)
    assert (got.method, got.handover) == ("scipy", "vol floor at 0 binds")
