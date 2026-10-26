"""
am_checks.py -- every numerical claim in metric_loading, checked with PASS/FAIL.
Runtime ~3-5 min on a laptop.  Usage:  python am_checks.py
Each check names the equation/section of the paper it supports and which solver(s) it uses:
  [TM] exact transfer matrix (am_model.z_*),  [FD] finite-difference strip (am_model.fd_*).
"""
import numpy as np
from scipy.optimize import brentq
from am_model import (chi_star, z_matched, z_general, classify, sigma_max_infinite,
                      fd_spectrum, fd_kappa2_negative, single_row_pencil, fd_pencil,
                      hw, V_P, C_PRIME, A_CELL, Z0, DX)
from scipy.linalg import eig

results = []
def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""))

GHz = lambda chi_si: V_P*chi_si/(2*np.pi)/1e9

# ---------------------------------------------------------------------------------------------
print("\n=== 1. Threshold equation and its limits (Sec. III, Eqs. 13, 16, 17) ===")
R = 0.05; c = chi_star(R); approx = (1 + 2*np.exp(-2/R))/R
check("weak-loading limit chi* ~ (1+2e^{-2a/r0})/r0", abs(c/approx - 1) < 1e-12, f"R={R}: {c:.12f} vs {approx:.12f}")
R = 100.0; c = chi_star(R)
check("strong-loading limit chi* ~ (r0 a)^{-1/2}", abs(c*np.sqrt(R) - 1) < 0.01, f"R={R}: chi* sqrt(R) = {c*np.sqrt(R):.4f}")
Rs = np.logspace(-2, 2, 200); cs = np.array([chi_star(r) for r in Rs])
check("chi* decreases monotonically with r0 ('stronger loading is more stable')", np.all(np.diff(cs) < 0))
for R in [0.15, 1.0, 3.0]:
    c = chi_star(R)
    check(f"band edges at R={R}: z(1/r0)=+1, z(chi*)=-1  [TM]",
          abs(z_matched(1/R, R) - 1) < 1e-12 and abs(z_matched(c, R) + 1) < 1e-9)

# ---------------------------------------------------------------------------------------------
print("\n=== 2. Negative-energy modes: count, location, energy sign (Secs. II-III, App. A) ===")
for R, m in [(0.15, 50), (3.0, 30)]:
    k2 = fd_kappa2_negative(R, M=20, m=m); chi = np.sqrt(-k2)
    check(f"R={R}: #(kappa^2<0) = #negative rows = 10  [FD]", len(k2) == 10, f"found {len(k2)}")
    check(f"R={R}: all negative-energy modes inside [1/R, chi*] (FD discretization tol 0.3%)  [FD]",
          chi.min() > (1/R)*(1-3e-3) and chi.max() < chi_star(R)*(1+3e-3),
          f"chi in [{chi.min():.4f}, {chi.max():.4f}], band [{1/R:.4f}, {chi_star(R):.4f}]")
# energy sign = sign of kappa^2: check <phi,W phi> < 0 for a kappa^2<0 eigenvector
A, B, K, wC = fd_pencil(3.0, 1.0, 1.0, 0.0, M=10, m=30)
lam, V = eig(K, B); ok = np.isfinite(lam); lam, V = lam[ok].real, V[:, ok].real
i_neg = np.argmin(lam); i_pos = np.argmin(np.where(lam > 0, lam, np.inf))
nW = lambda v: v @ (wC*v)
check("kappa^2<0 mode has negative capacitance-weighted norm; kappa^2>0 mode positive (Eq. 8)  [FD]",
      nW(V[:, i_neg]) < 0 and nW(V[:, i_pos]) > 0)

# ---------------------------------------------------------------------------------------------
print("\n=== 3. Matched strip: threshold k_x >= chi*, no oscillatory modes, growth rate (Eqs. 14-15) ===")
R = 3.0; c = chi_star(R)
for kx in [0.3, 0.99*c, 1.01*c, 2.0, 5.0]:
    nz, nc, s = fd_spectrum(R, 1, 1, kx)
    expect_unstable = kx < c
    check(f"gamma=1, kx={kx:.4f} (chi*={c:.4f}): {'unstable' if expect_unstable else 'stable'}, no complex modes  [FD]",
          (nz > 0) == expect_unstable and nc == 0, f"#zero-freq={nz}, #osc={nc}")
kx = 0.5*c
s_inf = sigma_max_infinite(R, 1.0, kx); s_th = np.sqrt(c**2 - kx**2)
check("sigma_max = v sqrt(chi*^2 - kx^2) on the infinite strip  [TM]", abs(s_inf/s_th - 1) < 1e-4,
      f"{s_inf:.6f} vs {s_th:.6f}")
s_fd = [fd_spectrum(R, 1, 1, kx, M=M, m=20)[2] for M in (10, 20, 40)]
check("finite strips approach sigma_max from below (infinite criterion is conservative)  [FD]",
      all(np.diff(s_fd) > 0) and s_fd[-1] < s_th, f"M=10,20,40: {np.round(s_fd, 4)}  -> {s_th:.4f}")

# ---------------------------------------------------------------------------------------------
print("\n=== 4. Ordinary negative capacitors, gamma = 0 (Sec. IV) ===")
R = 3.0
for kx in [0.5, 1.0, 3.0, 10.0]:
    nz, nc, s = fd_spectrum(R, 0, 0, kx)
    check(f"gamma=0, kx={kx}: unstable with one growing mode per negative row (8)  [FD]", nz == 8 and nc == 0,
          f"#zero-freq={nz}, #osc={nc}, growth={s:.3f}")
for r0, kx in [(0.5, 1.0), (0.5, 4.0)]:
    lam = single_row_pencil(r0, 0.0, kx, Lbox=12.0, m=60)
    s_fd = np.sqrt(-lam.real[lam.real < 0].min())
    s_th = np.sqrt((1 + np.sqrt(1 + 4*r0**2*kx**2))/(2*r0**2))
    check(f"isolated ordinary row, r0={r0}, kx={kx}: sigma^2 = [1+sqrt(1+4r0^2kx^2)]/(2r0^2) (Eq. 20)  [FD]",
          abs(s_fd/s_th - 1) < 5e-3, f"FD {s_fd:.4f} vs formula {s_th:.4f}")
g = [sigma_max_infinite(3.0, 0.0, k) for k in (0.5, 1.0, 2.0)]
check("gamma=0 growth increases as the strip narrows  [TM]", g[0] < g[1] < g[2], f"{np.round(g, 4)}")

# ---------------------------------------------------------------------------------------------
print("\n=== 5. Partial matching, gamma <= 1: threshold chi*(gamma r0) (Eq. 21) ===")
R = 3.0
for gam in [0.75, 0.5, 0.25]:
    th = chi_star(gam*R)
    lo = fd_spectrum(R, gam, gam, 0.99*th); hi = fd_spectrum(R, gam, gam, 1.01*th)
    zl = classify(R, gam, gam, 0.99*th); zh = classify(R, gam, gam, 1.01*th)
    check(f"gamma={gam}: unstable at 0.99 chi*(gR), stable at 1.01 chi*(gR), never oscillatory  [FD+TM]",
          lo[0] > 0 and lo[1] == 0 and hi[0] == 0 and hi[1] == 0 and zl == (True, False) and zh == (False, False),
          f"chi*(gR)={th:.4f}; FD below {lo[:2]}, above {hi[:2]}; TM below {zl}, above {zh}")
    y = np.linspace(0, 200, 200001); zz = z_general(-y, th, R, gam, gam)
    check(f"gamma={gam}: z decreases monotonically along x<0 at threshold (App. B)  [TM]", np.all(np.diff(zz) <= 1e-12))

# ---------------------------------------------------------------------------------------------
print("\n=== 6. Over-matching, gamma > 1: 4 r0^2 kx^2 (gamma-1) <= 1 (Eq. 23) ===")
for r0, gam in [(0.15, 1.05), (0.15, 1.2)]:
    kc = 1/(2*r0*np.sqrt(gam - 1))
    nb = np.sum(np.abs(single_row_pencil(r0, gam, 0.95*kc).imag) > 1e-6)
    na = np.sum(np.abs(single_row_pencil(r0, gam, 1.05*kc).imag) > 1e-6)
    check(f"isolated row r0={r0}, gamma={gam}: complex pair appears between 0.95 and 1.05 of kx_c={kc:.3f}  [FD]",
          nb == 0 and na == 2, f"#complex below={nb}, above={na}")
R = 3.0
def zmax_below_cutoff(gam, kx):
    x = np.linspace(1e-9, kx**2, 200001); return z_general(x, kx, R, gam, gam).max()
for kx in [0.62, 0.6974, 1.0, 1.5]:
    gc = brentq(lambda g: zmax_below_cutoff(g, kx) - 1, 1.0, 1.5)
    check(f"lattice R=3, kx={kx}: onset gamma_c = 1 + 1/(4R^2kx^2), lattice-independent  [TM]",
          abs(gc - (1 + 1/(4*R**2*kx**2))) < 1e-8, f"{gc:.10f} vs {1 + 1/(4*R**2*kx**2):.10f}")
# opposite energy signs of the colliding pair (isolated over-matched row, just below onset)
r0, gam = 0.15, 1.05; kx = 14.0
n = int(6.0*80) - 1; h = 1/80
Kd = (np.diag(2*np.ones(n)) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1))/h
wC = h*np.ones(n); wL = h*np.ones(n); i0 = n//2; wC[i0] -= 2*r0; wL[i0] -= 2*gam*r0
lam, V = eig(Kd + kx**2*np.diag(wL), np.diag(wC)); ok = np.isfinite(lam); lam, V = lam[ok].real, V[:, ok].real
roots = np.sort(np.roots([r0**2, -(2*r0**2*gam*kx**2 - 1), r0**2*gam**2*kx**4 - kx**2]).real)
signs = [np.sign(V[:, np.argmin(abs(lam - x))] @ (wC*V[:, np.argmin(abs(lam - x))])) for x in roots]
check("over-matched row binds a negative- and a positive-energy mode below cutoff (they collide)  [FD+TM]",
      sorted(signs) == [-1, 1] and all(x < kx**2 for x in roots), f"roots x={np.round(roots, 3)}, energy signs {signs}")

# ---------------------------------------------------------------------------------------------
print("\n=== 7. Positive vs negative rows separately (Sec. IV) ===")
R, kx = 3.003, 0.6974
zf, osc = classify(R, 0.95, 1.0, kx); nz, nc, _ = fd_spectrum(R, 0.95, 1.0, kx, M=24)
check("under-matched positive rows (gamma+=0.95, gamma-=1) oscillate  [TM+FD]", osc and nc > 0, f"TM osc={osc}, FD #osc={nc}")
def box_stable(cp, cn, dp, dn, npts=15):
    for gp in np.linspace(cp-dp, cp+dp, npts):
        for gn in np.linspace(cn-dn, cn+dn, npts):
            if any(classify(R, gp, gn, kx, n=30001)): return False
    return True
check("nominal matching: +-0.5% independent errors stable  [TM]", box_stable(1, 1, 0.005, 0.005))
check("nominal matching: +-1% independent errors reach instability  [TM]", not box_stable(1, 1, 0.01, 0.01))
check("biased design (1.2, 0.95): +-0.15 in gamma+, +-0.10 in gamma- all stable  [TM]", box_stable(1.2, 0.95, 0.15, 0.10))
def inductor_box_stable(cp, cn, ep, en, npts=9):
    """Relative errors in the series inductors: L = (L_L/c)(1+e)  ->  gamma = c/(1+e)."""
    return all(not any(classify(R, cp/(1+a), cn/(1+b), kx, n=30001))
               for a in np.linspace(-ep, ep, npts) for b in np.linspace(-en, en, npts))
check("biased design: +-12.5% passive-row and +-10% NIC-row inductor errors all stable  [TM]",
      inductor_box_stable(1.2, 0.95, 0.125, 0.10))
check("biased design: +-15% passive / +-10% NIC inductor errors is NOT fully stable (bound is tight)  [TM]",
      not inductor_box_stable(1.2, 0.95, 0.15, 0.10))
corners = [(1.05, 0.85), (1.05, 1.05), (1.35, 0.85), (1.35, 1.05), (1.2, 0.95)]
fd_ok = all(fd_spectrum(R, gp, gn, kx, M=24)[:2] == (0, 0) for gp, gn in corners)
check("biased design: box corners and centre stable on a finite strip  [FD]", fd_ok)

# ---------------------------------------------------------------------------------------------
print("\n=== 8. Hardware numbers (Table II, Sec. V) ===")
table = {0.5: (600.0, 12.73, 5.24), 2: (159.1, 3.37, 19.8), 5: (82.7, 1.76, 38.0),
         10: (55.1, 1.17, 57.1), 20: (37.8, 0.80, 83.1), 50: (23.5, 0.50, 133.6)}
for CL, (c_tab, nu_tab, W_tab) in table.items():
    r0, R_, c = hw(CL)
    ok = abs(c - c_tab) < 0.06 and abs(GHz(c) - nu_tab) < 0.006 and abs(np.pi/c*1e3 - W_tab) < 0.06
    check(f"Table II row C_L={CL} pF", ok, f"chi*={c:.2f}/m, nu*={GHz(c):.3f} GHz, Wmax={np.pi/c*1e3:.2f} mm, r0/a={R_:.3f}")
r0, R_, c = hw(10); kx = np.pi/50e-3
check("v_p = 1/(Z0 C') and L' = Z0^2 C' = 375 nH/m", abs(1/(Z0*C_PRIME)/V_P - 1) < 1e-3 and abs(Z0**2*C_PRIME - 375e-9) < 1e-12)
check("W_x=50 mm: kx=62.8/m, margin 14%, cutoff 1.33 GHz",
      abs(kx - 62.83) < 0.01 and abs(kx/c - 1 - 0.141) < 0.002 and abs(GHz(kx) - 1.333) < 0.001)
lo, hi = GHz(np.sqrt(kx**2 - c**2)), GHz(np.sqrt(kx**2 - 1/r0**2))
check("negative-energy modes occupy 0.64-1.17 GHz", abs(lo - 0.642) < 0.001 and abs(hi - 1.171) < 0.001, f"{lo:.3f}-{hi:.3f} GHz")
LL = (DX/V_P)**2/10e-12
check("matched L_L = 0.141 nH; biased 0.117 / 0.148 nH", abs(LL*1e9 - 0.1407) < 1e-3
      and abs(LL/1.2*1e9 - 0.117) < 1e-3 and abs(LL/0.95*1e9 - 0.148) < 1e-3)
Kd = 2/DX*np.sin(kx*DX/2)
check("discrete-row correction to kx is -0.4%", abs(Kd/kx - 1 + 0.0041) < 2e-4, f"{Kd/kx-1:+.4f}")
CLmin = brentq(lambda C: hw(C)[2] - kx, 0.3, 200)
check("threshold test: stable iff C_L >= 7.92 pF at W_x = 50 mm", abs(CLmin - 7.922) < 0.005, f"{CLmin:.3f} pF")
gmin = brentq(lambda g: chi_star(g*R_) - kx*A_CELL, 0.5, 1.0); gmax = 1 + 1/(4*R_**2*(kx*A_CELL)**2)
check("uniform matching window 0.79 <= gamma <= 1.057", abs(gmin - 0.792) < 0.001 and abs(gmax - 1.057) < 0.001,
      f"[{gmin:.4f}, {gmax:.4f}]")
fosc = GHz(np.sqrt(kx**2 - 1/(4*r0**2)))
check("over-match oscillation onset near 1.29 GHz", abs(fosc - 1.294) < 0.001, f"{fosc:.4f} GHz")
check("open-edge growth sigma = v chi* = 7.3e9 /s", abs(V_P*c/7.34e9 - 1) < 0.005, f"{V_P*c:.3e} /s")
check("negative row with W_x=50 mm, Delta_x=5 mm: 9 shunt + 10 series elements", round(50/5) - 1 == 9 and round(50/5) == 10)

# ---------------------------------------------------------------------------------------------
print("\n=== 9. Loss and NIC gain (Sec. VI, Eq. 25) ===")
from am_loss import shares, lossy_omegas, W_A
kx1 = np.pi/W_A
s1 = shares(kx1); ng, ps = s1[s1[:, 1] < 0], s1[s1[:, 1] > 0]
Lam = ng[:, 2] + ng[:, 3]
check("participation ratio of the band modes, order 1: Lambda = 3.6-7.8  [FD]",
      abs(Lam.min() - 3.6) < 0.05 and abs(Lam.max() - 7.8) < 0.05, f"{Lam.min():.2f}-{Lam.max():.2f}")
lo, hi = (ng[:, 2]/ng[:, 3]).max(), (ps[:, 2]/ps[:, 3]).min()
check("first-order NIC gain window 0.77 < rho < 1.37  [FD]", abs(lo - 0.772) < 0.005 and abs(hi - 1.375) < 0.005,
      f"{lo:.3f} < rho < {hi:.3f}")
w = lossy_omegas(kx1, 0.01, 0.01, 0.01); i = np.argmax(w.imag)
from am_loss import GHz as GHz_a
check("uniform Q=100: largest growth 4.6e8 /s near 1.14 GHz (direct lossy solve)  [FD]",
      abs(V_P*w.imag[i]/A_CELL/4.57e8 - 1) < 0.01 and abs(GHz_a(w.real[i]) - 1.14) < 0.01,
      f"{V_P*w.imag[i]/A_CELL:.3e} /s at {GHz_a(w.real[i]):.3f} GHz")
st = {r: lossy_omegas(kx1, 0.01, 0.01, -r*0.01).imag.max() <= 1e-9 for r in [0.7, 0.8, 1.3, 1.4]}
check("direct lossy solve: rho=0.7 unstable, 0.8 and 1.3 stable, 1.4 unstable  [FD]",
      st == {0.7: False, 0.8: True, 1.3: True, 1.4: False}, str(st))

# ---------------------------------------------------------------------------------------------
print(f"\n{sum(results)}/{len(results)} checks passed.")
