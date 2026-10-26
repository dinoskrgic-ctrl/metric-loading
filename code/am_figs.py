"""Figures for the metric-loading manuscript (no titles; panels carry only (a)-(c) labels). Run after am_maps.py (needs am_infinite_map.npz, am_tolerance_map.npz).
fig_krein_band.pdf, fig_phase.pdf, fig_matching.pdf"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa
from matplotlib.colors import ListedColormap
from am_model import chi_star, z_general, sigma_max_infinite

plt.style.use(["science"])
if "--no-latex" in sys.argv:          # for machines without a LaTeX install (cm-super): mathtext with CM fonts
    plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
                         "font.serif": ["cmr10", "DejaVu Serif"], "axes.formatter.use_mathtext": True})
BLUE, RED, GREY = "#1f5fa8", "#c0392b", "#777777"
RAMP = ["#0b2e59", "#1f5fa8", "#4a90c8"]
R_HW, KXA_HW = 3.003, 0.6974          # C_L = 10 pF, W_x = 50 mm
R_OLD, KXA_OLD = 0.150, 7.388         # C_L = 0.5 pF, W_x = 4.72 mm

# ---------- Fig. 2: z(i chi) vs u = r0 chi ----------
fig, ax = plt.subplots(figsize=(3.4, 2.5))
u = np.linspace(0.0, 2.6, 4000)
for col, R in zip(RAMP, [0.15, 1.0, 3.0]):
    z = 1 + 2*np.sinh(u/R)**2*(1 - u**2)
    ax.plot(u, z, color=col, lw=1.1, label=fr"$R={R}$")
    us = R*chi_star(R)
    ax.axvspan(1, us, color=col, alpha=0.18, lw=0)
ax.axhline(1, color=GREY, lw=0.5, ls="--"); ax.axhline(-1, color=GREY, lw=0.5, ls="--")
ax.set_ylim(-3, 3); ax.set_xlim(0, 2.6)
ax.set_xlabel(r"$u=r_0\chi$"); ax.set_ylabel(r"$z(i\chi)$")
ax.text(1.24, 2.3, "negative-energy\nbands", fontsize=6, color=GREY)
ax.legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig("fig_krein_band.pdf")

# ---------- Fig. 3: phase diagram ----------
fig, ax = plt.subplots(figsize=(3.4, 2.6))
R = np.logspace(-2, 2, 400); cs = np.array([chi_star(r) for r in R])
ax.fill_between(R, cs, 1e3, color=BLUE, alpha=0.15, lw=0)
ax.fill_between(R, 1e-3, cs, color=RED, alpha=0.15, lw=0)
ax.loglog(R, cs, color="k", lw=1.1, label=r"$\chi_*a$ (exact)")
ax.loglog(R, 1/R, color=GREY, lw=0.7, ls="--", label=r"$1/R$")
ax.loglog(R, R**-0.5, color=GREY, lw=0.7, ls=":", label=r"$R^{-1/2}$")
ax.plot([R_OLD], [KXA_OLD], "o", ms=4, mfc="white", mec="k", mew=0.8)
ax.plot([R_HW], [KXA_HW], "s", ms=4, color="k")
ax.annotate("0.5 pF", (R_OLD, KXA_OLD), xytext=(4, 4), textcoords="offset points", fontsize=6)
ax.annotate("10 pF", (R_HW, KXA_HW), xytext=(4, 4), textcoords="offset points", fontsize=6)
ax.text(6, 0.02, "unstable", fontsize=7, color=RED); ax.text(6, 15, "stable", fontsize=7, color=BLUE)
ax.set_xlim(1e-2, 1e2); ax.set_ylim(1e-2, 1e2)
ax.set_xlabel(r"$R=r_0/a$"); ax.set_ylabel(r"$k_x a$")
ax.legend(fontsize=6, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig("fig_phase.pdf")

# ---------- Fig. 4: matching ----------
d = np.load("am_infinite_map.npz"); G, KX, code = d["G"], d["KX"], d["code"]; Rm = float(d["R"])
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4))
cmap = ListedColormap(["#d6e4f3", "#f3d6d3", "#f7ead0"])
axs[0].pcolormesh(G, KX, code.T, cmap=cmap, vmin=0, vmax=2, shading="auto", rasterized=True)
g1 = np.linspace(0.02, 1.5, 300); lo = np.array([chi_star(g*Rm) for g in g1])
axs[0].plot(g1, lo, color="k", lw=1.0)
g2 = np.linspace(1.0005, 1.5, 300)
axs[0].plot(g2, 1/(2*Rm*np.sqrt(g2-1)), color="k", lw=1.0, ls="--")
axs[0].plot([1.0], [KXA_HW], "s", ms=3.5, color="k")
axs[0].set_ylim(0, 2); axs[0].set_xlim(0, 1.5)
axs[0].text(0.05, 0.2, "zero-frequency\ngrowth", fontsize=6, color=RED)
axs[0].text(0.5, 1.2, "stable", fontsize=7, color=BLUE)
axs[0].text(1.17, 0.72, "oscillatory\ngrowth", fontsize=6, color="#a0661a")
axs[0].text(0.27, 1.75, r"$\chi_*(\gamma r_0)$", fontsize=6)
axs[0].text(1.16, 1.78, r"$[2r_0\sqrt{\gamma-1}]^{-1}$", fontsize=5.5)
axs[0].set_xlabel(r"matching fraction $\gamma=\gamma_\pm$"); axs[0].set_ylabel(r"$k_x a$")
axs[0].text(0.04, 0.96, "(a)", transform=axs[0].transAxes, va="top", fontsize=8,
            bbox=dict(fc="white", ec="none", pad=0.6))

t = np.load("am_tolerance_map.npz"); GP, GN, tc = t["GP"], t["GN"], t["code"]
axs[1].pcolormesh(GP, GN, tc.T, cmap=cmap, vmin=0, vmax=2, shading="auto", rasterized=True)
axs[1].plot([1.0], [1.0], "s", ms=3.5, color="k")
axs[1].plot([1.2], [0.95], "D", ms=3.5, color="k")
axs[1].add_patch(plt.Rectangle((1.05, 0.85), 0.30, 0.20, fill=False, lw=0.8, ls="-", ec="k"))
axs[1].plot([0.6, 1.4], [0.6, 1.4], color=GREY, lw=0.5, ls=":")
axs[1].set_xlim(0.6, 1.4); axs[1].set_ylim(0.6, 1.4)
axs[1].set_xlabel(r"positive rows $\gamma_+$"); axs[1].set_ylabel(r"negative rows $\gamma_-$")
axs[1].text(0.04, 0.96, "(b)", transform=axs[1].transAxes, va="top", fontsize=8,
            bbox=dict(fc="white", ec="none", pad=0.6))
axs[1].text(0.63, 0.66, "zero-freq.", fontsize=6, color=RED)
axs[1].text(0.66, 1.25, "oscillatory", fontsize=6, color="#a0661a")
axs[1].text(0.9, 0.81, "stable", fontsize=7, color=BLUE)
ax2 = axs[2]
kk = np.linspace(0.0, 2.0, 121)
for col, g, lab in zip(["#c0392b", "#e08e45", BLUE], [0.0, 0.5, 1.0],
                       [r"$\gamma=0$ (ordinary $-C$)", r"$\gamma=0.5$", r"$\gamma=1$ (matched)"]):
    ax2.plot(kk, [sigma_max_infinite(Rm, g, k) for k in kk], color=col, lw=1.1, label=lab)
ax2.axvline(KXA_HW, color=GREY, lw=0.5, ls=":")
ax2.set_xlabel(r"$k_x a$"); ax2.set_ylabel(r"$\sigma_{\max}a/v_p$")
ax2.text(0.04, 0.96, "(c)", transform=ax2.transAxes, va="top", fontsize=8)
ax2.legend(fontsize=5.5, frameon=False, loc="lower right")
ax2.set_xlim(0, 2); ax2.set_ylim(0, 1.0)
fig.tight_layout(); fig.savefig("fig_matching.pdf")
print("saved")
