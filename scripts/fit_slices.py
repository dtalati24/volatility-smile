"""Fit every saved slice of one index in data/: spline base, bells on top, SVI for comparison.

    .venv/Scripts/python scripts/fit_slices.py                    # SPX, 3 knots
    .venv/Scripts/python scripts/fit_slices.py --knots 5
    .venv/Scripts/python scripts/fit_slices.py --underlying rut

Prints the fit errors and bell fit times per expiry (fast solver, and scipy for
comparison) and writes plots/fit_<underlying>.png: market mid vols
with bid/ask bars, the spline base, spline + bells and SVI; below each, the
errors in vol points.
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from volsmile import bells, spline, svi  # noqa: E402
from volsmile.calibration import fit_bells, fit_slice  # noqa: E402
from volsmile.marketdata import load_slice  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--knots", type=int, default=3)
parser.add_argument("--underlying", default="spx", help="file prefix in data/: spx or rut")
opts = parser.parse_args()
knots, under = opts.knots, opts.underlying

ROOT = Path(__file__).resolve().parent.parent
files = sorted((ROOT / "data").glob(f"{under}_*.csv"))
fig, ax = plt.subplots(2, len(files), figsize=(6 * len(files), 8), squeeze=False,
                       gridspec_kw={"height_ratios": [3, 1]})
if not files:
    raise SystemExit(f"no data/{under}_*.csv files")
print(f"{under.upper()}: RMSE in vol points, spline with {knots} knots")
print(f"{'expiry':10s} {'quotes':>6s} {'spline':>7s} {'+bells':>7s} {'SVI':>7s} {'fast ms':>8s} {'scipy ms':>9s}   buttons (vol pts)")
for j, path in enumerate(files):
    s = load_slice(path)
    q = s.quotes
    K = q["strike"].to_numpy()
    fit = fit_slice(s, knots=knots)
    grid = np.linspace(K.min(), K.max(), 600)
    base_grid = spline.vol(grid, s.forward, fit.base)
    base_q = spline.vol(K, s.forward, fit.base)
    total_q = bells.strike_vol(K, s.forward, s.tau, base_q, fit.buttons, fit.cfg)
    svi_text = "n/a" if fit.svi is None else f"{100 * fit.rmse_svi:.3f}"
    buttons = ", ".join(f"{name} {100 * v:+.2f}" for name, v in vars(fit.buttons).items())

    def best_ms(method, repeats=5):
        times = []
        for _ in range(repeats):
            started = time.perf_counter()
            fit_bells(s, base_q, fit.cfg, method)
            times.append(1000.0 * (time.perf_counter() - started))
        return min(times)
    print(f"{s.expiry:10s} {len(q):6d} {100 * fit.rmse_base:7.3f} {100 * fit.rmse_total:7.3f} {svi_text:>7s} "
          f"{best_ms('fast'):8.1f} {best_ms('scipy', 1):9.0f}   {buttons}")

    a = ax[0, j]
    a.vlines(K, 100 * q["iv_bid"], 100 * q["iv_ask"], color="#999999", lw=1, label="bid/ask")
    a.plot(K, 100 * q["iv"], ".", color="#1f2328", ms=4, label="mid")
    if fit.svi is not None:
        a.plot(grid, 100 * svi.vol(grid, s.forward, s.tau, fit.svi), color="#1e9e5a", lw=1.2,
               label=f"SVI, comparison (rmse {100 * fit.rmse_svi:.2f})")
    a.plot(grid, 100 * base_grid, color="#3b74d4", lw=1.5, ls="--", label=f"spline base (rmse {100 * fit.rmse_base:.2f})")
    a.plot(grid, 100 * bells.strike_vol(grid, s.forward, s.tau, base_grid, fit.buttons, fit.cfg),
           color="#d4763b", lw=1.5, label=f"spline + bells (rmse {100 * fit.rmse_total:.2f})")
    a.axvline(s.forward, color="#1f2328", lw=0.5)
    a.set_title(f"{under.upper()} {s.expiry} (tau {s.tau:.3f}), vol %")
    a.set_ylim(0, min(100, 100 * q["iv"].max() * 1.15))
    a.legend(frameon=False, fontsize=8)
    b = ax[1, j]
    b.plot(K, 100 * (base_q - q["iv"]), ".", color="#3b74d4", ms=4, label="spline - mid")
    b.plot(K, 100 * (total_q - q["iv"]), ".", color="#d4763b", ms=4, label="spline + bells - mid")
    b.axhline(0, color="#1f2328", lw=0.6)
    b.set_ylabel("error (vol pts)")
    b.legend(frameon=False, fontsize=8)
    for x in (a, b):
        x.grid(alpha=0.25)
fig.tight_layout()
(ROOT / "plots").mkdir(exist_ok=True)
fig.savefig(ROOT / "plots" / f"fit_{under}.png", dpi=100)
