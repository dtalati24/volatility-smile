from dataclasses import astuple, fields
from statistics import NormalDist

import numpy as np
import pytest
from scipy.optimize import brentq

from volsmile import black76
from volsmile.bells import DELTAS, NODES, BellConfig, BellParams, strike_vol, vol_change, widths

NAMES = [f.name for f in fields(BellParams)]
U = {d: NormalDist().inv_cdf(d / 100) for d in DELTAS}  # put side, negative
P = BellParams(put_1=0.006, put_2=0.004, put_5=0.0025, put_10=0.0015, put_25=0.0005, atm=-0.003,
               call_25=-0.004, call_10=0.002, call_5=0.01, call_2=0.002, call_1=0.003)

# An SPX-like one-month slice with a smooth base smile.
F, TAU, ATM = 7670.65, 0.0973, 0.1405
K = np.linspace(0.8, 1.1, 121) * F
Z = np.log(K / F) / (ATM * np.sqrt(TAU))
BASE = ATM - 0.03 * Z + 0.012 * Z**2


def one(name, value=0.01):
    return BellParams(**{name: value})


def mirror(p):
    swap = {n: getattr(p, n.replace("put", "CALL").replace("call", "put").replace("CALL", "call")) for n in NAMES}
    return BellParams(**swap)


def bell(u, centre, width):
    return np.exp(-((u - centre) ** 2) / (2 * width**2))


def gap(vol, strike, base, p):
    """Self-consistency: zero when vol is the model vol at this strike."""
    u = -black76.d1(F, strike, TAU, vol)
    return max(0.0, base + float(vol_change(u, p))) - vol


def test_defaults():
    assert BellConfig().k == 0.4
    assert DELTAS == (1, 2, 5, 10, 25)
    assert astuple(BellParams()) == (0.0,) * 11


def test_nodes_are_the_delta_points_in_field_order():
    assert U[25] == pytest.approx(-0.6744897501960817, rel=1e-14)
    assert U[1] == pytest.approx(-2.3263478740408408, rel=1e-14)
    expected = [U[d] for d in DELTAS] + [0.0] + [-U[d] for d in reversed(DELTAS)]
    np.testing.assert_array_equal(NODES, expected)
    assert NAMES == [f"put_{d}" for d in DELTAS] + ["atm"] + [f"call_{d}" for d in reversed(DELTAS)]


def test_widths_by_hand():
    k = 0.4
    put = [
        k * (U[2] - U[1]),                      # tip: only one neighbour
        k * (U[5] - U[1]) / 2,
        k * (U[10] - U[2]) / 2,
        k * (U[25] - U[5]) / 2,
        k * (0.0 - U[10]) / 2,
    ]
    atm = k * (-2 * U[25]) / 2
    np.testing.assert_allclose(widths(), put + [atm] + put[::-1], rtol=1e-14)
    np.testing.assert_allclose(widths(BellConfig(k=1.0)), widths() / 0.4, rtol=1e-14)


def test_no_buttons_adds_nothing():
    u = np.linspace(-8, 8, 161)
    np.testing.assert_array_equal(vol_change(u, BellParams()), np.zeros_like(u))
    np.testing.assert_allclose(strike_vol(K, F, TAU, BASE, BellParams()), BASE, rtol=1e-14)


@pytest.mark.parametrize("i, name", list(enumerate(NAMES)))
def test_a_button_adds_exactly_its_value_at_its_node_and_leaks_to_neighbours(i, name):
    got = vol_change(NODES, one(name))
    assert got[i] == 0.01
    np.testing.assert_allclose(got, 0.01 * bell(NODES, NODES[i], widths()[i]), rtol=1e-12, atol=1e-300)


def test_vol_change_matches_the_formula_and_mirrors():
    u = np.linspace(-4, 4, 401)
    held = [np.maximum(u, NODES[0])] + [u] * 9 + [np.minimum(u, NODES[-1])]  # tips hold past their node
    expected = sum(b * bell(x, c, w) for b, x, c, w in zip(astuple(P), held, NODES, widths()))
    np.testing.assert_allclose(vol_change(u, P), expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(vol_change(-u, mirror(P)), vol_change(u, P), rtol=1e-12, atol=1e-15)


def test_k_changes_only_the_width():
    wide = BellConfig(k=0.8)
    assert vol_change(NODES[4], one("put_25"), wide) == 0.01
    assert vol_change(NODES[3], one("put_25"), wide) > vol_change(NODES[3], one("put_25"))


@pytest.mark.parametrize("name, d, is_call", [("put_25", 25, False), ("put_5", 5, False), ("call_10", 10, True)])
def test_button_lands_exactly_on_the_true_delta_of_the_final_smile(name, d, is_call):
    # Flat base 0.2 and one button b: at the strike whose final-smile delta is
    # d, the vol is exactly 0.2 + b.
    tau, vol = 0.25, 0.2 + 0.01
    s = vol * np.sqrt(tau)
    target_d1 = -U[d] if not is_call else U[d]          # put: -N(-d1) = -d  |  call: N(d1) = d
    strike = 100.0 * np.exp(-(target_d1 - s / 2) * s)
    got = float(strike_vol(strike, 100.0, tau, 0.2, one(name)))
    assert got == pytest.approx(vol, rel=1e-13)
    assert abs(float(black76.delta(100.0, strike, tau, got, is_call))) == pytest.approx(d / 100, rel=1e-12)


def test_atm_button_sits_at_50_delta_just_above_the_forward():
    tau, vol = 1.0, 0.25
    strike = 100.0 * np.exp(vol**2 * tau / 2)            # d1 = 0
    assert float(strike_vol(strike, 100.0, tau, 0.2, one("atm", 0.05))) == pytest.approx(vol, rel=1e-13)
    assert float(strike_vol(100.0, 100.0, tau, 0.2, one("atm", 0.05))) < vol


def test_every_strike_is_self_consistent():
    got = strike_vol(K, F, TAU, BASE, P)
    residual = [gap(v, k, b, P) for v, k, b in zip(got, K, BASE)]
    assert np.max(np.abs(residual)) < 1e-14
    assert np.max(np.abs(got - BASE)) > 0.005                   # the bells did something


def test_matches_an_independent_solve():
    for strike, base in zip(K[::20], BASE[::20]):
        expected = brentq(gap, 1e-6, 2.0, args=(strike, base, P), xtol=1e-15, rtol=1e-15)
        assert float(strike_vol(strike, F, TAU, base, P)) == pytest.approx(expected, rel=1e-13)


def test_solves_where_simple_repetition_would_not_settle():
    # 0.75 vol point buttons of alternating sign: repeating vol -> delta -> vol
    # never settles on this slice, but the solution is unique and found.
    alternating = BellParams(**{n: 0.0075 * (-1) ** i for i, n in enumerate(NAMES)})
    got = strike_vol(K, F, TAU, BASE, alternating)
    residual = [gap(v, k, b, alternating) for v, k, b in zip(got, K, BASE)]
    assert np.max(np.abs(residual)) < 1e-14


def test_more_than_one_self_consistent_vol_raises():
    # +-1 vol point alternating is already ambiguous: at 1.075 F the vols
    # 0.1160, 0.1204 and 0.1288 are all self-consistent.
    alternating = BellParams(**{n: 0.01 * (-1) ** i for i, n in enumerate(NAMES)})
    with pytest.raises(ValueError, match="more than one"):
        strike_vol(K, F, TAU, BASE, alternating)
    strike = 1.075 * F
    base = float(np.interp(strike, K, BASE))
    for vol in (0.11601, 0.12042, 0.12876):
        lo, hi = gap(vol - 2e-5, strike, base, alternating), gap(vol + 2e-5, strike, base, alternating)
        assert lo * hi < 0


def test_a_second_solution_at_zero_vol_raises():
    # At K = F vol 0 can be self-consistent (the ATM bell pulls it to 0) while
    # a positive vol is too: solutions 0, 0.2889 and 0.8427.
    with pytest.raises(ValueError, match="more than one"):
        strike_vol(100.0, 100.0, 4.0, 0.2, BellParams(atm=-0.3, put_25=0.8))


def test_buttons_land_on_the_true_deltas_with_many_buttons_and_a_sloped_base():
    def base(strike):
        # Gentler than BASE, so every node's delta is reached within the range searched.
        z = np.log(strike / F) / (ATM * np.sqrt(TAU))
        return ATM - 0.015 * z + 0.003 * z**2

    for i, (name, d_true) in enumerate(zip(NAMES, [-0.01, -0.02, -0.05, -0.10, -0.25, None, 0.25, 0.10, 0.05, 0.02, 0.01])):
        def off_node(log_k):
            strike = F * np.exp(log_k)
            vol = float(strike_vol(strike, F, TAU, base(strike), P))
            return -float(black76.d1(F, strike, TAU, vol)) - NODES[i]

        strike = F * np.exp(brentq(off_node, -0.45, 0.25, xtol=1e-14))
        vol = float(strike_vol(strike, F, TAU, base(strike), P))
        assert vol - base(strike) == pytest.approx(float(vol_change(NODES[i], P)), abs=1e-13)
        if d_true is not None:
            assert float(black76.delta(F, strike, TAU, vol, d_true > 0)) == pytest.approx(d_true, rel=1e-11)


def test_base_delta_option_places_bells_at_the_base_smiles_delta():
    cfg = BellConfig(delta="base")
    alternating = BellParams(**{n: 0.01 * (-1) ** i for i, n in enumerate(NAMES)})
    got = strike_vol(K, F, TAU, BASE, alternating, cfg)
    expected = BASE + vol_change(-black76.d1(F, K, TAU, BASE), alternating)
    np.testing.assert_allclose(got, expected, rtol=1e-14)
    # At the base smile's 25 delta put strike the button adds exactly its value.
    tau, s = 0.25, 0.2 * np.sqrt(0.25)
    strike = 100.0 * np.exp(-(-U[25] - s / 2) * s)
    assert float(strike_vol(strike, 100.0, tau, 0.2, one("put_25"), cfg)) == pytest.approx(0.21, rel=1e-13)


def test_inner_bells_fade_back_to_the_base_far_from_the_money():
    far = np.array([0.3, 3.0]) * F
    inner = BellParams(**{n: getattr(P, n) for n in NAMES[1:-1]})
    np.testing.assert_allclose(strike_vol(far, F, TAU, 0.3, inner), [0.3, 0.3], rtol=1e-14)


@pytest.mark.parametrize("cfg", [BellConfig(), BellConfig(delta="base")])
def test_one_delta_bells_hold_their_button_out_to_the_far_wings(cfg):
    far = np.array([0.3, 0.5, 2.0, 3.0]) * F
    np.testing.assert_allclose(strike_vol(far, F, TAU, 0.3, P, cfg), 0.3 + np.array([P.put_1] * 2 + [P.call_1] * 2),
                               rtol=1e-14)
    # Held flat past the node, a bell inside it: no jump, no kink at the node.
    for i, name in ((0, "put_1"), (10, "call_1")):
        side = -1.0 if i == 0 else 1.0
        u = NODES[i] + side * np.array([-1e-6, 0.0, 1e-6, 1.0])
        got = vol_change(u, one(name))
        np.testing.assert_allclose(got[1:], 0.01, rtol=1e-14)
        assert 0.01 - got[0] < 1e-12


def test_negative_vol_floors_at_zero():
    assert float(strike_vol(100.0, 100.0, 1.0, 0.2, one("atm", -0.5))) == 0.0


def test_scalar_base_broadcasts_and_shapes():
    got = strike_vol(np.array([90.0, 100.0, 110.0]), 100.0, 1.0, 0.2, BellParams())
    np.testing.assert_allclose(got, [0.2] * 3, rtol=1e-14)
    assert np.ndim(strike_vol(100.0, 100.0, 1.0, 0.2, P)) == 0
    assert strike_vol(np.full((4, 2), 100.0), 100.0, 1.0, np.full((4, 2), 0.2), P).shape == (4, 2)
    assert strike_vol(np.empty(0), 100.0, 1.0, 0.2, P).shape == (0,)


@pytest.mark.parametrize("kwargs", [{"k": 0.0}, {"k": -0.4}, {"k": np.nan}, {"k": np.inf}, {"delta": "strike"}])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        BellConfig(**kwargs)


@pytest.mark.parametrize("args", [
    (-5.0, 100.0, 1.0, 0.2),                        # strike
    (100.0, 0.0, 1.0, 0.2),                         # forward
    (100.0, 100.0, 0.0, 0.2),                       # tau
    ([90.0, 100.0], 100.0, 1.0, [0.2, 0.2, 0.2]),   # base does not match strikes
    (100.0, 100.0, 1.0, [0.2, 0.2]),                # one strike, several base vols
    ([90.0, 100.0], 100.0, 1.0, [0.2, np.nan]),     # base not finite
    (100.0, 100.0, 1.0, 0.0),                       # base not > 0
])
def test_bad_inputs(args):
    with pytest.raises(ValueError):
        strike_vol(*args, P)


@pytest.mark.parametrize("cfg", [BellConfig(), BellConfig(delta="base")])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_buttons_raise(cfg, bad):
    with pytest.raises(ValueError, match="buttons"):
        strike_vol([90.0, 110.0], 100.0, 1.0, 0.2, BellParams(put_25=bad), cfg)
