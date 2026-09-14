"""Bells at delta nodes that add straight onto the vol smile.

One bell per node: 1, 2, 5, 10, 25 delta puts, ATM, and the same for calls.
Bell = exp(-(z - node)^2 / (2 width^2)), height 1 at its node, so a button of
+1 adds 1 vol point there. Nodes are placed in z = log(K/F) / (atm vol sqrt(t)),
where a put of delta d sits at z = N^-1(d). The nodes are deliberately bunched
at the tips (finer control there).

Figure 1 (delta_bells_widths.png): one row per bell width.
  left:   every bell, and their sum when all buttons are +1 (flat = no ripple)
  middle: vol change from three single buttons: 25P, 5P, 1C at +1
  right:  base smile and the smile with those three buttons applied
  last row: width measured in delta instead of z, for comparison
Figure 3 (delta_bells_k.png): each bell gets its own width, k x the average
distance to its neighbouring nodes, so tip bells are narrow and middle bells
wide. All bells plain (a one-sided last bell was tried and rejected).
k defaults to 0.4; compare several with --k 0.3 0.4 0.5. Titles show the condition number of the
node matrix (1 = buttons independent, large = combinations of buttons that
barely change the smile).
Figure 2 (delta_bells_points.png): leakage onto neighbouring nodes, and what
happens past the last node (plain bell vs one-sided last bell).
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=float, nargs="+", default=[0.4],
                    help="bell width = k x distance to neighbouring nodes; one row per value (default 0.4)")
KS = parser.parse_args().k

PLOTS = os.path.join(os.path.dirname(__file__), "..", "plots")
DELTAS = [1, 2, 5, 10, 25]
NAMES = [f"{d}P" for d in DELTAS] + ["ATM"] + [f"{d}C" for d in reversed(DELTAS)]
NODES = np.array([norm.ppf(d / 100) for d in DELTAS] + [0.0] + [-norm.ppf(d / 100) for d in reversed(DELTAS)])
WIDTHS = [0.1, 0.2, 0.35, 0.6]
GAPS = np.diff(NODES)
NEIGHBOUR = np.concatenate([[GAPS[0]], (GAPS[:-1] + GAPS[1:]) / 2, [GAPS[-1]]])
DELTA_WIDTH = 0.05
BLUE, ORANGE, GREEN, INK = "#3b74d4", "#d4763b", "#5aa469", "#1f2328"
BUMPS = {"25P": BLUE, "5P": ORANGE, "1C": GREEN}

z = np.linspace(-3.2, 3.2, 2001)


def base(z):
    return 20.0 - 0.8 * z + 1.2 * z**2


def bells(z, width, in_delta=False, one_sided=False):
    """11 x len(z) array, one bell per node. width: one number or one per node."""
    x, c = (norm.cdf(z), norm.cdf(NODES)) if in_delta else (z, NODES)
    w = np.broadcast_to(width, NODES.shape)[:, None]
    out = np.exp(-((x[None, :] - c[:, None]) ** 2) / (2 * w**2))
    if one_sided:  # outermost bells stay at 1 beyond their node
        out[0, z < NODES[0]] = 1.0
        out[-1, z > NODES[-1]] = 1.0
    return out


def style(a, ticks=True):
    a.grid(alpha=0.25)
    a.axhline(0, color=INK, lw=0.6)
    if ticks:
        a.set_xticks(NODES, NAMES, fontsize=7, rotation=90)


def width_figure(rows, filename):
    fig, ax = plt.subplots(len(rows), 3, figsize=(16, 3.4 * len(rows)), squeeze=False)
    for r, (w, in_delta, one_sided, label) in enumerate(rows):
        B = bells(z, w, in_delta, one_sided)
        for b in B:
            ax[r, 0].plot(z, b, color=BLUE, lw=1, alpha=0.6)
        ax[r, 0].plot(z, B.sum(axis=0), color=INK, lw=2, ls="--", label="sum, all buttons +1")
        ax[r, 0].set_title(f"{label}: bells and their sum")
        ax[r, 0].legend(frameon=False, fontsize=8)
        bumped = base(z)
        for name, colour in BUMPS.items():
            change = B[NAMES.index(name)]
            ax[r, 1].plot(z, change, color=colour, lw=2, label=f"{name} +1")
            bumped = bumped + change
        ax[r, 1].set_title("vol change from single buttons")
        ax[r, 1].legend(frameon=False, fontsize=8)
        ax[r, 2].plot(z, base(z), color=INK, lw=1.5, label="base")
        ax[r, 2].plot(z, bumped, color=ORANGE, lw=2, label="25P, 5P, 1C all +1")
        ax[r, 2].set_title("smile (vol %)")
        ax[r, 2].legend(frameon=False, fontsize=8)
        for a in ax[r]:
            style(a)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, filename), dpi=100)


def condition(width):
    """Condition number of the matrix of bell heights at the nodes."""
    return np.linalg.cond(bells(NODES, width))


width_figure([(w, False, False, f"width {w} in z (cond {condition(w):.1f})") for w in WIDTHS]
             + [(DELTA_WIDTH, True, False, f"width {DELTA_WIDTH} in delta")], "delta_bells_widths.png")
width_figure([(k * NEIGHBOUR, False, False, f"k = {k}: widths {k * NEIGHBOUR.min():.2f}-{k * NEIGHBOUR.max():.2f} (cond {condition(k * NEIGHBOUR):.1f})")
              for k in KS], "delta_bells_k.png")

# Figure 2: leakage (point 4) and past the last node (point 5)
W = 0.2
zt = np.linspace(-4.2, 0.3, 1501)
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
change = bells(zt, W)[NAMES.index("5P")]
ax[0].plot(zt, change, color=ORANGE, lw=2.5, label="5P button +1")
for name, c in zip(NAMES[:6], NODES[:6]):
    v = np.exp(-((c - NODES[NAMES.index("5P")]) ** 2) / (2 * W**2))
    ax[0].plot(c, v, "o", color=INK, ms=7)
    ax[0].annotate(f"{name}: +{v:.2f}", (c, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax[0].set_title(f"Point 4, leakage (width {W}): 5P +1 moves 5P by exactly 1,\nbut also moves the neighbouring nodes")
plain = base(zt) + 2 * bells(zt, W)[0]
sided = base(zt) + 2 * bells(zt, W, one_sided=True)[0]
ax[1].plot(zt, base(zt), color=INK, lw=1.5, label="base")
ax[1].plot(zt, plain, color=BLUE, lw=2, label="1P +2, plain bell: far wing falls back to base")
ax[1].plot(zt, sided, color=ORANGE, lw=2, ls="--", label="1P +2, one-sided last bell: stays +2")
ax[1].set_title("Point 5, past the last node (1P)")
ax[1].set_ylabel("vol (%)")
far = {"0.01P": norm.ppf(1e-4), "0.1P": norm.ppf(1e-3)}
for a in ax:
    style(a, ticks=False)
    a.set_xticks(list(far.values()) + list(NODES[:6]), list(far) + NAMES[:6], fontsize=8, rotation=90)
    a.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "delta_bells_points.png"), dpi=100)
