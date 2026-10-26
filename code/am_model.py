"""
am_model.py -- core model for "Metric loading" (metric_loading).

Units: row spacing a = 1, host velocity v_p = 1, so x = (omega/v_p)^2 and kx are in units of 1/a.
R = r0/a with r0 = C_L / (2 C').

Wave equation (Eq. 19 of the paper), rows at y = n a with signed strength r_n = s_n r0:
    -phi'' + kx^2 W_gamma phi = x W phi,
    W       = 1 + 2 sum_n r_n delta(y - n a)            (capacitance profile = metric)
    W_gamma = 1 + 2 sum_n gamma_n r_n delta(y - n a)     (series-inductance loading)
gamma_n = 1: matched (covariant) row.  gamma_n = 0: ordinary capacitor, no series element.

Two independent solvers:
  (A) exact transfer matrix of the doubled cell  -> half-trace z(x)        [functions z_*]
  (B) finite-difference generalized eigenproblem on a finite Dirichlet strip [fd_pencil, fd_spectrum]
(B) never uses (A); agreement between them is the verification.
"""
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.linalg import eig, eigh

# ---------------------------------------------------------------- closed forms
def chi_star(R):
    """Root of (R^2 chi^2 - 1) sinh^2(chi) = 1 above 1/R (Eq. 13), units 1/a."""
    f = lambda c: ((R*c)**2 - 1)*np.sinh(c)**2 - 1
    lo = 1.0/R                       # f(1/R) = -1 exactly
    hi = lo*(1 + 1e-6)
    while f(hi) < 0:
        hi = lo + 2*(hi - lo)
    return brentq(f, lo, hi, xtol=1e-14)

def z_matched(chi, R):
    """Half-trace at kappa = i chi, matched loading (Eq. 12)."""
    return 1 + 2*np.sinh(chi)**2*(1 - (R*chi)**2)

def z_general(x, kx, R, gp, gn):
    """Exact doubled-cell half-trace for decay mu = sqrt(kx^2 - x) (Appendix B):
       z = cosh 2mu + (g1+g2) sinh(2mu)/(2mu) + g1 g2 sinh^2(mu)/(2 mu^2),
       g1 = +2R(gp kx^2 - x) (positive row), g2 = -2R(gn kx^2 - x) (negative row).
       Written with sinh(mu)/mu so it is analytic through mu = 0 and valid for x > kx^2."""
    x = np.asarray(x, float)
    mu = np.sqrt((kx**2 - x).astype(complex))
    small = np.abs(mu) < 1e-8
    mu_safe = np.where(small, 1.0, mu)
    s1 = np.where(small, 1.0, np.sinh(mu)/mu_safe)          # sinh(mu)/mu
    s2 = np.where(small, 2.0, np.sinh(2*mu)/mu_safe)        # sinh(2mu)/mu
    g1 = 2*R*(gp*kx**2 - x)
    g2 = -2*R*(gn*kx**2 - x)
    return (np.cosh(2*mu) + (g1 + g2)*s2/2 + g1*g2*s1**2/2).real

def classify(R, gp, gn, kx, n=60001):
    """Infinite strip, from the exact half-trace.
       zero_freq: some x < 0 with |z| <= 1  (real growth, Re w = 0)
       osc:       a stationary point of z(x), x > 0, with |z| < 1 (complex x pair -> oscillatory growth).
       Extrema are refined with a bounded optimizer so the tangential Bragg touchings (z = 1 exactly)
       are not mistaken for interior extrema."""
    xn = -np.linspace(0, 60 + 4*kx**2 + 40/R**2, n)[1:]
    zero_freq = bool(np.any(np.abs(z_general(xn, kx, R, gp, gn)) <= 1))
    xp = np.linspace(1e-9, kx**2 + 60, n)
    zp = z_general(xp, kx, R, gp, gn)
    dz = np.diff(zp)
    turns = np.where(np.sign(dz[1:]) != np.sign(dz[:-1]))[0] + 1
    osc = False
    for i in turns:
        sg = 1.0 if zp[i] > zp[i-1] else -1.0
        r = minimize_scalar(lambda t: -sg*z_general([t], kx, R, gp, gn)[0],
                            bounds=(xp[i-1], xp[i+1]), method="bounded", options={"xatol": 1e-13})
        if abs(-sg*r.fun) < 1 - 1e-7:
            osc = True
            break
    return zero_freq, osc

def sigma_max_infinite(R, g, kx):
    """Largest zero-frequency growth rate (units v_p/a), uniform matching g, infinite strip."""
    y = np.linspace(0, 30 + 4*kx**2 + 40/R**2, 300001)
    ok = np.abs(z_general(-y, kx, R, g, g)) <= 1
    return np.sqrt(y[ok].max()) if ok.any() else 0.0

# ---------------------------------------------------------------- finite-difference solver
def fd_pencil(R, gp, gn, kx, M=16, m=30):
    """Dirichlet strip [0, M+1], M rows at y = 1..M alternating +,-,+,... starting with -.
       Quadratic forms: int phi'^2 + kx^2 int W_gamma phi^2  vs  x int W phi^2.
       Returns (A, B, K, wC): A u = x B u, K = -d^2/dy^2 (FD), wC = diag of B."""
    L = M + 1; n = L*m - 1; h = 1.0/m
    K = (np.diag(2*np.ones(n)) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1))/h
    wC = h*np.ones(n); wL = h*np.ones(n)
    for j in range(1, M + 1):
        r = (-1)**j*R
        g = gn if r < 0 else gp
        i = j*m - 1
        wC[i] += 2*r; wL[i] += 2*g*r
    return K + kx**2*np.diag(wL), np.diag(wC), K, wC

def fd_spectrum(R, gp, gn, kx, M=16, m=30):
    """Eigenvalues x of the finite strip; returns (#zero-frequency unstable, #oscillatory unstable, max growth)."""
    A, B, _, _ = fd_pencil(R, gp, gn, kx, M, m)
    lam = eig(A, B, right=False)
    lam = lam[np.isfinite(lam)]
    cplx = np.abs(lam.imag) > 1e-6*np.maximum(1, np.abs(lam.real))
    neg = (lam.real < -1e-9) & ~cplx
    bad = lam[cplx | neg]
    growth = np.sqrt(-bad.astype(complex)).real.max() if len(bad) else 0.0   # Im(omega) in units v_p/a
    return int(neg.sum()), int(cplx.sum()), growth

def fd_kappa2_negative(R, M=20, m=40):
    """Matched loading: kappa^2 spectrum of (-d^2, W) via the symmetric form K^{-1/2} W K^{-1/2}
       (K > 0 with Dirichlet ends). Returns the negative kappa^2 values."""
    _, _, K, wC = fd_pencil(R, 1.0, 1.0, 0.0, M, m)
    ev, U = eigh(K)
    Kih = U @ np.diag(ev**-0.5) @ U.T
    mu = eigh(Kih @ np.diag(wC) @ Kih, eigvals_only=True)   # = 1/kappa^2
    return 1/mu[mu < 0]

def single_row_pencil(r0, g, kx, Lbox=6.0, m=80):
    """One negative row in a box: returns eigenvalues x."""
    n = int(Lbox*m) - 1; h = 1.0/m
    K = (np.diag(2*np.ones(n)) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1))/h
    wC = h*np.ones(n); wL = h*np.ones(n); i = n//2
    wC[i] -= 2*r0; wL[i] -= 2*g*r0
    lam = eig(K + kx**2*np.diag(wL), np.diag(wC), right=False)
    return lam[np.isfinite(lam)]

# ---------------------------------------------------------------- hardware (SI)
V_P, C_PRIME, A_CELL, Z0, DX = 1.333e8, 150e-12, 11.1e-3, 50.0, 5e-3
def hw(CL_pF):
    """r0 [m], R, chi* [1/m] for a load C_L in pF."""
    r0 = CL_pF*1e-12/(2*C_PRIME); R = r0/A_CELL
    return r0, R, chi_star(R)/A_CELL
