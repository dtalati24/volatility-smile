"""Smile board: a local web app for the spline + bells model on saved index slices (SPX, RUT).

    .venv/Scripts/python app/server.py            # then open http://127.0.0.1:8050
    .venv/Scripts/python app/server.py --port 9000

Python does all the maths; the page (app/static) only draws and sends button
values. Standard library HTTP server, bound to 127.0.0.1 only.

API (JSON):
    GET  /api/slices                 saved slices in data/
    GET  /api/fit?slice=NAME&knots=3&k=0.4&delta=final
                                     Fit: spline base + bells fitted to the mids
                                     (SVI alongside for comparison), the smile and
                                     board for it, and how the fit went
    POST /api/autofit                {slice, base_slice, knots, fitted: {...},
                                      offsets: {...}, k, delta, stiffness,
                                      rmse_at_fit (optional), stale_ratio, stale_button}
                                     automatic fitter: the spline from base_slice
                                     kept fixed, bells refitted to the slice's mids
                                     starting from `fitted`; the smile drawn with
                                     fitted + offsets; stale-spline reasons
    POST /api/smile                  {slice, base_slice, knots, buttons: {...}, k, delta}
                                     the smile and board for given buttons on
                                     base_slice's spline base (default: slice)
Errors come back as {"error": message} with status 400 (bad input), 403 (a Host
other than 127.0.0.1 or localhost), 404, 413 (body over 1 MB) or 500.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from volsmile import black76, bells, spline, svi
from volsmile.calibration import STALE_BUTTON, STALE_ERROR_RATIO, fit_base, fit_bells, fit_svi, rmse, stale_reasons
from volsmile.marketdata import load_slice

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STATIC = Path(__file__).resolve().parent / "static"
GRID_POINTS = 400
MAX_KNOTS = 30
MAX_BODY = 1_000_000
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}


def slice_names() -> list[str]:
    return sorted(p.stem for p in DATA.glob("*_*.csv"))


@lru_cache(maxsize=None)
def get_slice(name: str):
    if name not in slice_names():  # also stops paths like ../../x
        raise LookupError(f"no saved slice called {name!r}")
    return load_slice(DATA / f"{name}.csv")


@lru_cache(maxsize=None)
def get_base(name: str, knots: int) -> spline.Spline:
    return fit_base(get_slice(name), knots)


@lru_cache(maxsize=None)
def get_svi(name: str):
    return fit_svi(get_slice(name))


def number(value, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def config(k, delta) -> bells.BellConfig:
    return bells.BellConfig(k=number(k, "k"), delta=str(delta))


def knot_count(value) -> int:
    if isinstance(value, bool):
        raise ValueError("knots must be a whole number")
    n = number(value, "knots")
    if n != int(n) or not 0 <= n <= MAX_KNOTS:
        raise ValueError(f"knots must be a whole number from 0 to {MAX_KNOTS}")
    return int(n)


def buttons_from(value, name: str) -> bells.BellParams:
    """Buttons from a JSON object; missing ones are 0."""
    b = {} if value is None else value
    if not isinstance(b, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = set(b) - set(bells.BellParams.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown {name}: {sorted(unknown)}")
    return bells.BellParams(**{n: number(v, n) for n, v in b.items()})


def base_name_for(name: str, base_name) -> str:
    """The slice whose spline is used: the same index and expiry as `name`."""
    base_name = name if base_name in (None, "") else str(base_name)
    s, b = get_slice(name), get_slice(base_name)
    if base_name.split("_")[0] != name.split("_")[0] or b.expiry != s.expiry:
        raise ValueError(f"base_slice {base_name!r} is not the same index and expiry as {name!r}")
    return base_name


def smile(name: str, knots: int, buttons: bells.BellParams, cfg: bells.BellConfig, base_name: str | None = None) -> dict:
    """Curves for the chart and one row per quote for the board. The spline
    base is the one fitted to base_name (default: this slice), tied to
    moneyness, so it moves with this slice's forward."""
    base_name = base_name_for(name, base_name)
    s = get_slice(name)
    q = s.quotes
    K = q["strike"].to_numpy()
    base = get_base(base_name, knots)
    p_svi = get_svi(name)
    grid = np.linspace(K.min(), K.max(), GRID_POINTS)
    base_grid = spline.vol(grid, s.forward, base)
    base_q = spline.vol(K, s.forward, base)
    svi_grid = np.full(grid.shape, np.nan) if p_svi is None else svi.vol(grid, s.forward, s.tau, p_svi)
    svi_q = None if p_svi is None else svi.vol(K, s.forward, s.tau, p_svi)
    model_grid = bells.strike_vol(grid, s.forward, s.tau, base_grid, buttons, cfg)
    model_q = bells.strike_vol(K, s.forward, s.tau, base_q, buttons, cfg)
    is_call = q["is_call"].to_numpy()
    price_q = black76.price(s.forward, K, s.tau, model_q, is_call)
    delta_q = black76.delta(s.forward, K, s.tau, np.maximum(model_q, 1e-12), is_call)
    mid = q["iv"].to_numpy()

    # Strike of each node: where u = -d1 crosses it on the grid, using the vol the
    # bells are placed with (the model's own vol, or the base smile's).
    placing = base_grid if cfg.delta == "base" else model_grid
    u_grid = -black76.d1(s.forward, grid, s.tau, np.maximum(placing, 1e-12))
    nodes = []
    for node_name, u_node in zip(bells.BellParams.__dataclass_fields__, bells.NODES):
        cross = np.flatnonzero(np.diff(np.sign(u_grid - u_node)) != 0)
        if cross.size:
            i = cross[0]
            t = (u_node - u_grid[i]) / (u_grid[i + 1] - u_grid[i])
            nodes.append({"name": node_name, "strike": float(grid[i] + t * (grid[i + 1] - grid[i]))})

    def clean(a):
        return [None if not np.isfinite(x) else float(x) for x in np.asarray(a, dtype=float)]

    return {
        "slice": {"name": name, "expiry": s.expiry, "as_of": s.as_of.isoformat(), "forward": s.forward, "tau": s.tau},
        "base_slice": base_name,
        "knots": knots,
        "svi": None if p_svi is None else asdict(p_svi),
        "buttons": asdict(buttons),
        "k": cfg.k,
        "delta": cfg.delta,
        "rmse_base": rmse(base_q, mid),
        "rmse_model": rmse(model_q, mid),
        "rmse_svi": None if svi_q is None else rmse(svi_q, mid),
        "curve": {"strike": clean(grid), "base": clean(base_grid), "svi": clean(svi_grid), "model": clean(model_grid)},
        "nodes": nodes,
        "board": {
            "strike": clean(K), "is_call": [bool(c) for c in is_call],
            "bid": clean(q["bid"]), "ask": clean(q["ask"]), "mid": clean(q["mid"]),
            "iv_bid": clean(q["iv_bid"]), "iv_ask": clean(q["iv_ask"]), "iv_mid": clean(mid),
            "model_vol": clean(model_q), "model_price": clean(price_q), "model_delta": clean(delta_q),
            "volume": clean(q["volume"]), "open_interest": clean(q["open_interest"]),
        },
    }


def fit_report(b, seconds: float, rmse_fitted: float | None, spline_seconds: float | None = None) -> dict:
    report = {"method": b.method, "handover": b.handover, "converged": b.converged,
              "ms": 1000.0 * seconds, "rmse": rmse_fitted}
    if spline_seconds is not None:
        report["spline_ms"] = 1000.0 * spline_seconds
    return report


def fit(name: str, knots: int, cfg: bells.BellConfig) -> dict:
    """Fit: a fresh spline and fresh bells, fitted to the mids (no offsets)."""
    s = get_slice(name)
    K = s.quotes["strike"].to_numpy()
    started = time.perf_counter()
    base = fit_base(s, knots)                       # timed fresh; get_base holds the same fit for drawing
    spline_seconds = time.perf_counter() - started
    base_vol = spline.vol(K, s.forward, base)
    started = time.perf_counter()
    b = fit_bells(s, base_vol, cfg)
    seconds = time.perf_counter() - started
    out = smile(name, knots, b.buttons, cfg)
    return out | {"bells_converged": b.converged, "fitted": asdict(b.buttons),
                  "fit": fit_report(b, seconds, out["rmse_model"], spline_seconds)}


def autofit(body: dict) -> dict:
    """Automatic fitter: spline from base_slice kept fixed, bells refitted to this
    slice's mids from the previous fitted buttons; offsets are added on top
    only for drawing."""
    name = str(body.get("slice", ""))
    base_name = base_name_for(name, body.get("base_slice"))
    knots = knot_count(body.get("knots", 3))
    cfg = config(body.get("k", 0.4), body.get("delta", "final"))
    previous = buttons_from(body.get("fitted"), "fitted")
    offsets = buttons_from(body.get("offsets"), "offsets")
    stiffness = number(body.get("stiffness", 0.0), "stiffness")
    rmse_at_fit = None if body.get("rmse_at_fit") is None else number(body["rmse_at_fit"], "rmse_at_fit")
    ratio = number(body.get("stale_ratio", STALE_ERROR_RATIO), "stale_ratio")
    limit = number(body.get("stale_button", STALE_BUTTON), "stale_button")
    if stiffness < 0.0 or (rmse_at_fit is not None and rmse_at_fit < 0.0) or ratio <= 0.0 or limit <= 0.0:
        raise ValueError("stiffness and rmse_at_fit must be >= 0; stale_ratio and stale_button > 0")

    s = get_slice(name)
    K, mids = s.quotes["strike"].to_numpy(), s.quotes["iv"].to_numpy()
    base_vol = spline.vol(K, s.forward, get_base(base_name, knots))
    started = time.perf_counter()
    b = fit_bells(s, base_vol, cfg, start=previous, stiffness=stiffness)
    seconds = time.perf_counter() - started
    try:
        rmse_fitted = rmse(bells.strike_vol(K, s.forward, s.tau, base_vol, b.buttons, cfg), mids)
    except bells.AmbiguousVolError:  # the fallback fit stopped next to an ambiguous region
        rmse_fitted = float("inf")
    stale = stale_reasons(rmse_fitted, rmse_at_fit, b.buttons, b.handover, ratio, limit)
    total = bells.BellParams(*(f + o for f, o in zip(asdict(b.buttons).values(), asdict(offsets).values())))
    out = smile(name, knots, total, cfg, base_name)
    report = fit_report(b, seconds, rmse_fitted if np.isfinite(rmse_fitted) else None)
    return out | {"fitted": asdict(b.buttons), "offsets": asdict(offsets), "fit": report, "stale": stale}


def from_body(body: dict) -> dict:
    name = str(body.get("slice", ""))
    return smile(name, knot_count(body.get("knots", 3)), buttons_from(body.get("buttons"), "buttons"),
                 config(body.get("k", 0.4), body.get("delta", "final")), body.get("base_slice"))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, format, *args):  # quiet: no line per request
        pass

    def end_headers(self):
        # Static files: the browser must check for a newer copy, so an edited
        # app.js is not replaced by a cached one. (API replies send no-store.)
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload, allow_nan=False).encode()  # NaN/inf is not valid JSON: ValueError
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def run(self, work):
        try:
            self.send_json(work())
        except LookupError as e:
            self.send_json({"error": str(e)}, HTTPStatus.NOT_FOUND)
        except ValueError as e:
            self.send_json({"error": str(e) or "invalid input"}, HTTPStatus.BAD_REQUEST)
        except Exception as e:  # anything else still gets a reply, not a dropped connection
            self.send_json({"error": f"server error: {type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def host_ok(self):
        # Blocks DNS rebinding: a page on another site cannot talk to this server
        # through a hostname that resolves to 127.0.0.1.
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        if host in ALLOWED_HOSTS:
            return True
        self.send_json({"error": "forbidden host"}, HTTPStatus.FORBIDDEN)
        return False

    def do_GET(self):
        if not self.host_ok():
            return
        url = urlparse(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        if url.path == "/api/slices":
            self.run(slice_names)
        elif url.path == "/api/fit":
            self.run(lambda: fit(query.get("slice", ""), knot_count(query.get("knots", 3)),
                                 config(query.get("k", 0.4), query.get("delta", "final"))))
        elif url.path.startswith("/api/"):
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        else:
            super().do_GET()

    def do_POST(self):
        if not self.host_ok():
            return
        route = {"/api/smile": from_body, "/api/autofit": autofit}.get(urlparse(self.path).path)
        if route is None:
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY:
            self.send_json({"error": f"body must be 0 to {MAX_BODY} bytes"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            self.close_connection = True
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError
        except (ValueError, RecursionError):
            self.send_json({"error": "body must be a JSON object"}, HTTPStatus.BAD_REQUEST)
            return
        self.run(lambda: route(body))


def make_server(port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8050)
    server = make_server(parser.parse_args().port)
    print(f"Smile board on http://127.0.0.1:{server.server_address[1]}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
