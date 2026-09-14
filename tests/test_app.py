import json
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from volsmile import bells, spline, svi
from volsmile.calibration import fit_slice
from volsmile.marketdata import load_slice

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import server  # noqa: E402

NAME = "spx_2026-10-16_2026-09-11"


@pytest.fixture(scope="module")
def base_url():
    srv = server.make_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url):
    try:
        with urlopen(url) as r:
            return r.status, r.read()
    except HTTPError as e:
        return e.code, e.read()


def post(url, payload, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode()
    try:
        with urlopen(Request(url, data=data, headers={"Content-Type": "application/json"})) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_and_static_files_are_served(base_url):
    for path, text in (("/", b"Smile Board"), ("/app.js", b"press"), ("/style.css", b".knob")):
        status, body = get(base_url + path)
        assert status == 200 and text in body
        with urlopen(base_url + path) as r:  # an edited page is picked up without a hard refresh
            assert r.headers.get_all("Cache-Control") == ["no-cache"]


def test_slices_lists_the_saved_files(base_url):
    status, body = get(base_url + "/api/slices")
    assert status == 200 and NAME in json.loads(body)


@pytest.mark.parametrize("knots", [3, 5])
def test_fit_matches_the_library(base_url, knots):
    status, body = get(f"{base_url}/api/fit?slice={NAME}&knots={knots}&k=0.4&delta=final")
    assert status == 200
    got = json.loads(body)
    expected = fit_slice(load_slice(ROOT / "data" / f"{NAME}.csv"), knots=knots)
    assert got["knots"] == knots and got["bells_converged"] == expected.bells_converged
    assert got["svi"] == pytest.approx(asdict(expected.svi), rel=1e-9)
    assert got["buttons"] == pytest.approx(asdict(expected.buttons), rel=1e-9, abs=1e-12)
    assert got["rmse_base"] == pytest.approx(expected.rmse_base, rel=1e-9)
    assert got["rmse_model"] == pytest.approx(expected.rmse_total, rel=1e-9)
    assert got["rmse_svi"] == pytest.approx(expected.rmse_svi, rel=1e-9)
    assert len(got["board"]["strike"]) == len(got["board"]["model_vol"]) > 100
    assert [n["name"] for n in got["nodes"]] == list(bells.BellParams.__dataclass_fields__)


def test_smile_uses_the_posted_buttons_on_the_spline_base(base_url):
    s = load_slice(ROOT / "data" / f"{NAME}.csv")
    q = s.quotes
    buttons = bells.BellParams(put_25=0.01, call_10=-0.005)
    status, got = post(base_url + "/api/smile", {"slice": NAME, "knots": 4, "buttons": asdict(buttons),
                                                 "k": 0.5, "delta": "base"})
    assert status == 200
    K = q["strike"].to_numpy()
    base = spline.vol(K, s.forward, spline.fit(K, s.forward, s.tau, q["iv"].to_numpy(), 4))
    expected = bells.strike_vol(K, s.forward, s.tau, base, buttons, bells.BellConfig(k=0.5, delta="base"))
    np.testing.assert_allclose(got["board"]["model_vol"], expected, rtol=1e-12)
    grid = np.array(got["curve"]["strike"])
    np.testing.assert_allclose(got["curve"]["base"], spline.vol(grid, s.forward, spline.fit(K, s.forward, s.tau,
                               q["iv"].to_numpy(), 4)), rtol=1e-12)
    p_svi = svi.fit(K, s.forward, s.tau, q["iv"].to_numpy())
    np.testing.assert_allclose(got["curve"]["svi"], svi.vol(grid, s.forward, s.tau, p_svi), rtol=1e-9)
    # Buttons left out of the request are 0, and knots default to 3.
    status, got = post(base_url + "/api/smile", {"slice": NAME, "buttons": {"atm": 0.01}})
    assert status == 200 and got["buttons"]["put_1"] == 0.0 and got["buttons"]["atm"] == 0.01 and got["knots"] == 3


@pytest.mark.parametrize("payload, status", [
    ({"slice": "../pyproject"}, 404),
    ({"slice": NAME, "buttons": {"put_3": 0.1}}, 400),
    ({"slice": NAME, "buttons": {"atm": "x"}}, 400),
    ({"slice": NAME, "k": -1}, 400),
    ({"slice": NAME, "knots": -1}, 400),
    ({"slice": NAME, "knots": 2.5}, 400),
    ({"slice": NAME, "knots": 31}, 400),
    ({"slice": NAME, "knots": "many"}, 400),
    ({"slice": NAME, "knots": True}, 400),
])
def test_bad_requests_return_an_error_message(base_url, payload, status):
    code, body = post(base_url + "/api/smile", payload)
    assert code == status and body["error"]


def test_ambiguous_vols_return_an_error_message(base_url):
    alternating = {n: 0.05 * (-1) ** i for i, n in enumerate(bells.BellParams.__dataclass_fields__)}
    code, body = post(base_url + "/api/smile", {"slice": NAME, "buttons": alternating})
    assert code == 400 and "more than one" in body["error"]


def test_malformed_body_and_unknown_routes(base_url):
    assert post(base_url + "/api/smile", None, raw=b"[1, 2]")[0] == 400
    assert post(base_url + "/api/smile", None, raw=b"{not json")[0] == 400
    assert get(base_url + "/api/nothing")[0] == 404
    assert get(f"{base_url}/api/fit?slice=nope")[0] == 404


def raw_request(base_url, method, path, headers, body=b""):
    import http.client

    host, port = base_url.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=10)
    conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    for k, v in headers.items():
        conn.putheader(k, v)
    conn.endheaders()
    if body:
        conn.send(body)
    r = conn.getresponse()
    status, data = r.status, r.read()
    conn.close()
    return status, data


def test_other_host_headers_are_refused(base_url):
    port = base_url.rsplit(":", 1)[1]
    assert raw_request(base_url, "GET", "/api/slices", {"Host": f"evil.example:{port}"})[0] == 403
    assert raw_request(base_url, "GET", "/api/slices", {"Host": f"localhost:{port}"})[0] == 200


@pytest.mark.parametrize("length", ["-1", "100000000000", "abc"])
def test_bad_content_length_is_refused(base_url, length):
    port = base_url.rsplit(":", 1)[1]
    status, _ = raw_request(base_url, "POST", "/api/smile", {"Host": f"127.0.0.1:{port}", "Content-Length": length})
    assert status == 413


@pytest.mark.parametrize("payload", [
    {"slice": NAME, "buttons": ["atm"]},
    {"slice": NAME, "buttons": 5},
    {"slice": NAME, "knots": [3]},
    {"slice": NAME, "buttons": {"atm": 1e300}, "delta": "base"},        # result not finite
])
def test_wrong_types_and_non_finite_results_get_a_400(base_url, payload):
    code, body = post(base_url + "/api/smile", payload)
    assert code == 400 and body["error"]


def test_deeply_nested_body_gets_a_400(base_url):
    assert post(base_url + "/api/smile", None, raw=b"[" * 100000 + b"]" * 100000)[0] == 400


def test_node_markers_follow_the_delta_mode(base_url):
    from volsmile import black76

    fitted = json.loads(get(f"{base_url}/api/fit?slice={NAME}")[1])
    s = load_slice(ROOT / "data" / f"{NAME}.csv")
    q = s.quotes
    base_fit = spline.fit(q["strike"].to_numpy(), s.forward, s.tau, q["iv"].to_numpy())
    for delta in ("final", "base"):
        _, got = post(base_url + "/api/smile", {"slice": NAME, "buttons": fitted["buttons"], "delta": delta})
        for n in got["nodes"]:
            k = n["strike"]
            base = float(spline.vol(k, s.forward, base_fit))
            vol = base if delta == "base" else float(
                bells.strike_vol(k, s.forward, s.tau, base, bells.BellParams(**fitted["buttons"])))
            u = -float(black76.d1(s.forward, k, s.tau, vol))
            node = bells.NODES[list(bells.BellParams.__dataclass_fields__).index(n["name"])]
            assert u == pytest.approx(node, abs=2e-3)


def test_fit_reports_how_the_fit_went(base_url):
    got = json.loads(get(f"{base_url}/api/fit?slice={NAME}")[1])
    assert got["fitted"] == got["buttons"]                       # Fit: no offsets
    report = got["fit"]
    assert (report["method"], report["handover"], report["converged"]) == ("fast", None, True)
    assert 0.0 < report["ms"] < 1000.0 and 0.0 < report["spline_ms"] < 100.0
    assert report["rmse"] == got["rmse_model"]
    assert got["base_slice"] == NAME


def test_autofit_keeps_the_spline_refits_bells_and_adds_offsets_on_top(base_url):
    from volsmile.calibration import fit_bells

    first = json.loads(get(f"{base_url}/api/fit?slice={NAME}")[1])
    offsets = {"put_10": 0.0025, "atm": -0.001}
    payload = {"slice": NAME, "base_slice": NAME, "knots": 3, "fitted": first["fitted"], "offsets": offsets,
               "k": 0.4, "delta": "final", "stiffness": 0.5, "rmse_at_fit": first["fit"]["rmse"]}
    status, got = post(base_url + "/api/autofit", payload)
    assert status == 200 and got["stale"] == [] and got["fit"]["method"] == "fast"

    s = load_slice(ROOT / "data" / f"{NAME}.csv")
    K, mids = s.quotes["strike"].to_numpy(), s.quotes["iv"].to_numpy()
    base = spline.vol(K, s.forward, spline.fit(K, s.forward, s.tau, mids, 3))
    expected = fit_bells(s, base, start=bells.BellParams(**first["fitted"]), stiffness=0.5).buttons
    assert got["fitted"] == pytest.approx(asdict(expected), abs=1e-12)
    total = {n: got["fitted"][n] + offsets.get(n, 0.0) for n in got["fitted"]}
    assert got["buttons"] == pytest.approx(total, abs=1e-15) and got["offsets"]["put_10"] == 0.0025
    np.testing.assert_allclose(got["board"]["model_vol"], bells.strike_vol(K, s.forward, s.tau, base,
                               bells.BellParams(**total)), rtol=1e-12)
    # The reported fit error is the bells' fit to the mids, without the offsets.
    no_offsets = bells.strike_vol(K, s.forward, s.tau, base, expected)
    assert got["fit"]["rmse"] == pytest.approx(float(np.sqrt(np.mean((no_offsets - mids) ** 2))), rel=1e-9)
    assert got["rmse_model"] > got["fit"]["rmse"]                 # the offsets move the drawn smile off the mids


def test_autofit_flags_a_stale_spline(base_url):
    first = json.loads(get(f"{base_url}/api/fit?slice={NAME}")[1])
    payload = {"slice": NAME, "fitted": first["fitted"], "rmse_at_fit": first["fit"]["rmse"] / 2}
    status, got = post(base_url + "/api/autofit", payload)
    assert status == 200 and len(got["stale"]) == 1 and "above 1.5x" in got["stale"][0]
    status, got = post(base_url + "/api/autofit", payload | {"stale_ratio": 3})
    assert status == 200 and got["stale"] == []
    status, got = post(base_url + "/api/autofit", payload | {"stale_ratio": 3, "stale_button": 0.001})
    assert status == 200 and "buttons beyond 0.1 vol pts" in got["stale"][0]
    status, got = post(base_url + "/api/autofit", {"slice": NAME, "fitted": first["fitted"]})   # no rmse_at_fit
    assert status == 200 and got["stale"] == []


@pytest.mark.parametrize("payload", [
    {"slice": NAME, "stiffness": -1},
    {"slice": NAME, "rmse_at_fit": -0.1},
    {"slice": NAME, "stale_ratio": 0},
    {"slice": NAME, "fitted": [0.01]},
    {"slice": NAME, "offsets": {"put_3": 0.01}},
    {"slice": NAME, "base_slice": "spx_2026-12-18_2026-09-11"},      # another expiry's spline
    {"slice": NAME, "base_slice": "rut_2026-10-16_2026-09-11"},      # another index's spline
])
def test_autofit_bad_requests_get_a_400(base_url, payload):
    code, body = post(base_url + "/api/autofit", payload)
    assert code == 400 and body["error"]


def test_smile_refuses_a_spline_from_another_expiry(base_url):
    code, body = post(base_url + "/api/smile", {"slice": NAME, "base_slice": "spx_2026-09-21_2026-09-11"})
    assert code == 400 and "same index and expiry" in body["error"]
