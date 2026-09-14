"""The earlier standalone bell model (flat ATM vol plus bells, no base smile)
beyond the 1 delta put, under different rules. Superseded: the bells are now
an overlay on a base smile, so far out the smile is the base smile.

Left: the smile. Right: the risk-neutral density implied by it (second
derivative of Black-76 put prices in strike). A negative density is an
arbitrage: a butterfly spread that costs less than nothing.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

from volsmile.bells import NODES as NODES_WITH_ATM, widths  # noqa: E402

F, TAU, ATM = 100.0, 0.25, 0.20
# The old model had no ATM bell; the other widths are unchanged.
NODES = np.delete(NODES_WITH_ATM, 5)
W = np.delete(widths(), 5)
buttons = np.array([0.10, 0.08, 0.05, 0.03, 0.01, 0, 0, 0, 0, 0])


def plain(z, outer=None):
    """Sum of bells, no rule at the tip; outer = width of the 1P bell on its far side."""
    w = np.tile(W, (len(z), 1))
    if outer is not None:
        w[z < NODES[0], 0] = outer
    return ATM + np.sum(buttons * np.exp(-((z[:, None] - NODES) ** 2) / (2 * w**2)), axis=1)


def straight_line(z):
    """The removed rule: beyond 1P, a line with the slope between the 2P and 1P vols."""
    at = plain(NODES[:2]) - ATM
    slope = (at[1] - at[0]) / (NODES[1] - NODES[0])
    inside = np.maximum(z, NODES[0])
    return plain(inside) + slope * (z - inside)


z = np.linspace(-4.5, 0.0, 9001)
rules = {
    "bells carried on (plain 1P bell)": plain(z),
    "1P bell wider on far side (width 1)": plain(z, 1.0),
    "1P bell wider on far side (width 3)": plain(z, 3.0),
    "straight line (removed rule)": straight_line(z),
}
colours = ["#d4763b", "#a05ec4", "#5aa469", "#3b74d4"]

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
K = F * np.exp(z * ATM * np.sqrt(TAU))
for (name, vol), c in zip(rules.items(), colours):
    s = vol * np.sqrt(TAU)
    d1 = np.log(F / K) / s + s / 2
    put = K * norm.cdf(s - d1) - F * norm.cdf(-d1)  # puts: accurate at low strikes
    density = np.gradient(np.gradient(put, K), K)
    ax[0].plot(z, 100 * vol, color=c, lw=2, label=name)
    ax[1].plot(z, density, color=c, lw=2, label=name)
    print(f"{name:40s} vol at 0.1P {100 * np.interp(norm.ppf(1e-3), z, vol):5.1f}  "
          f"at 0.01P {100 * np.interp(norm.ppf(1e-4), z, vol):5.1f}  min density {density[50:-50].min():+.5f} at z {z[50 + np.argmin(density[50:-50])]:.2f}")
ticks = [norm.ppf(d) for d in (1e-4, 1e-3, 0.01, 0.02, 0.05, 0.10, 0.25)] + [0.0]
labels = ["0.01P", "0.1P", "1P", "2P", "5P", "10P", "25P", "ATM"]
for a in ax:
    a.set_xticks(ticks, labels, fontsize=8, rotation=90)
    a.grid(alpha=0.25)
    a.legend(frameon=False, fontsize=8)
ax[0].set_title("Put wing (vol %)")
ax[1].set_title("Implied density (below 0 = arbitrage)")
ax[1].axhline(0, color="#1f2328", lw=0.8)
ax[1].set_ylim(-0.01, 0.03)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "..", "plots", "bells_beyond_1delta.png"), dpi=100)
