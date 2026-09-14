# Explorations

Design experiments for the bell overlay, kept so they can be rerun. Not part of
the package. Each writes a figure into `plots/` (gitignored), except where
noted. Run from the repo
root, e.g.

    .venv/Scripts/python explorations/delta_bells.py

| Script | What it shows |
|---|---|
| `delta_bells.py` | One Gaussian bell per delta node (1/2/5/10/25 puts and calls, ATM) added straight onto vol; compares fixed widths with per-node widths (k x distance to neighbours, `--k`, default 0.4), the leakage onto neighbouring nodes, and the far wing past the last node |
| `delta_modes.py` | Final vs spline vs frozen delta for the automatic fitter: the smile change on the first auto-fit tick after a Refit, and the shape of a 5Δ put click. Writes `docs/images/delta_modes.png` (committed, shown in the README) |
| `bells_beyond_1delta.py` | The earlier standalone bell model (flat ATM vol plus bells, no base smile) past the 1 delta put: bells carried on, the 1 delta bell wider on its far side, a straight line; with the implied density (negative = arbitrage) |

Why the design ended where it did is summarised in the README
("What we tried, and why we chose what we did").
