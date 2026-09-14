"""Which delta should the automatic fitter put the bells at? Refit uses the
final smile's delta; the candidates for the auto-fit ticks are:

- spline delta: positions from the spline alone (bells ignored);
- frozen delta: positions recorded from the final smile at the last Refit and
  kept until the next Refit.

Left: the first auto-fit tick after a Refit with no market move, as the change
in the smile. Right: a +2 vol point click on the 5 delta put under each model.
All three fit the mids equally well (printed). Writes docs/images/delta_modes.png,
which the README shows.
"""
from dataclasses import astuple, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from volsmile import bells, calibration, spline  # noqa: E402
from volsmile.marketdata import load_slice  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIN, SPL = bells.BellConfig(delta="final"), bells.BellConfig(delta="base")
SLICES = {"SPX 1 week": "spx_2026-09-21_2026-09-11", "SPX 1 month": "spx_2026-10-16_2026-09-11",
          "RUT 1 month": "rut_2026-10-16_2026-09-11"}
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"

fig, axes = plt.subplots(len(SLICES), 2, figsize=(13, 3.4 * len(SLICES)))
for row, (label, name) in zip(axes, SLICES.items()):
    s = load_slice(ROOT / "data" / f"{name}.csv")
    K, iv = s.quotes["strike"].to_numpy(), s.quotes["iv"].to_numpy()
    sp = calibration.fit_base(s)
    grid = np.linspace(K.min(), K.max(), 400)
    base_q, base_g = spline.vol(K, s.forward, sp), spline.vol(grid, s.forward, sp)
    x_fin = calibration.fit_bells(s, base_q, FIN).buttons
    x_spl = calibration.fit_bells(s, base_q, SPL).buttons

    def vol(strikes, base, x, cfg):
        return bells.strike_vol(strikes, s.forward, s.tau, base, x, cfg)

    refit = vol(grid, base_g, x_fin, FIN)
    rt = np.sqrt(s.tau)
    u_frozen = -(np.log(s.forward / grid) / (refit * rt) + 0.5 * refit * rt)
    frozen = lambda x: base_g + bells.bell_matrix(u_frozen, FIN) @ np.array(astuple(x))  # noqa: E731

    jump_kept = vol(grid, base_g, x_fin, SPL) - refit      # spline delta, Refit's buttons
    jump_refit = vol(grid, base_g, x_spl, SPL) - refit     # spline delta, buttons refitted
    jump_frozen = frozen(x_fin) - refit

    click = replace(x_fin, put_5=x_fin.put_5 + 0.02)
    add_fin = vol(grid, base_g, click, FIN) - refit
    add_spl = vol(grid, base_g, click, SPL) - vol(grid, base_g, x_fin, SPL)
    add_frz = frozen(click) - frozen(x_fin)

    err = {m: 100 * calibration.rmse(vol(K, base_q, x, cfg), iv)
           for m, x, cfg in (("final", x_fin, FIN), ("spline", x_spl, SPL))}
    print(f"{label}: fit error final {err['final']:.3f}, spline {err['spline']:.3f} vol pts | first tick, largest change: "
          f"spline kept {100 * np.abs(jump_kept).max():.3f}, spline refitted {100 * np.abs(jump_refit).max():.3f}, "
          f"frozen {100 * np.abs(jump_frozen).max():.1e} vol pts | 5P click peaks at "
          f"{grid[add_fin.argmax()]:.0f} final, {grid[add_spl.argmax()]:.0f} spline, {grid[add_frz.argmax()]:.0f} frozen")

    a = row[0]
    a.plot(grid, 100 * jump_kept, color=ORANGE, lw=2, ls="--", label="spline delta, Refit buttons kept")
    a.plot(grid, 100 * jump_refit, color=ORANGE, lw=1.5, ls=":", label="spline delta, buttons refitted")
    a.plot(grid, 100 * jump_frozen, color=GREEN, lw=2, label="frozen delta (from last Refit)")
    a.set_ylim(-0.25, 0.25)
    a.set_title(f"{label}: first auto-fit tick, no market move", fontsize=10)
    a.set_ylabel("smile change (vol pts)")

    a = row[1]
    a.plot(grid, 100 * add_fin, color=BLUE, lw=2, label="final delta")
    a.plot(grid, 100 * add_spl, color=ORANGE, lw=2, ls="--", label="spline delta")
    a.plot(grid, 100 * add_frz, color=GREEN, lw=2, ls=":", label="frozen delta")
    a.set_title(f"{label}: +2 vol point click on the 5Δ put", fontsize=10)
    a.set_ylabel("vol added (vol pts)")
    for a in row:
        a.axvline(s.forward, color="#898781", lw=0.8)
        a.grid(alpha=0.25)
        a.legend(frameon=False, fontsize=8, loc="upper left")
for a in axes[-1]:
    a.set_xlabel("strike (grey line: forward)")
fig.tight_layout()
out = ROOT / "docs" / "images" / "delta_modes.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=100)
print(f"wrote {out}")
