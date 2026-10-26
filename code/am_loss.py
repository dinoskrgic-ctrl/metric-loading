"""
am_loss.py -- effect of small losses (and of NIC gain) on the matched strip (Sec. VI of the paper).

Each element e carries a loss tangent d_e = 1/Q_e (d_e > 0 dissipative, d_e < 0 net gain), implemented
as C_e -> C_e + i|C_e| d_e and L_e -> L_e + i|L_e| d_e (time dependence e^{-i omega t}).
Element groups: host (lanes and links), positive rows (passive C and L), negative rows (NIC C and L).

(1) First order.  Dissipated power P = sum_e 2 omega |U_e| d_e.  A mode of signed energy U changes at
    dU/dt = -P, so its amplitude rate is sigma = -(omega/U) sum_e |U_e| d_e.  Negative-energy modes
    (U < 0) grow under dissipation; positive-energy modes decay.  With uniform loss, sigma = Lambda omega/Q,
    Lambda = sum_e |U_e| / |U| >= 1 (participation ratio).
(2) Direct check.  Complex generalized eigenproblem of the lossy finite strip (no perturbation theory).
(3) NIC gain.  With d_host = d_pos = d and d_NIC = -rho d, first order predicts that every mode decays iff
    max_{U<0} (Lam_P/Lam_N) < rho < min_{U>0} (Lam_P/Lam_N), Lam_P = host + positive-row share,
    Lam_N = NIC share.

Usage: python am_loss.py        (~10 min; writes am_loss_output.txt content to stdout)
"""
import numpy as np
from scipy.linalg import eig
from am_model import fd_pencil

V_P, A_CELL = 1.333e8, 11.1e-3
R_HW, W_A = 3.003, 50/11.1           # 10-pF design, W_x = 50 mm, in units of a
M, MGRID = 40, 30                     # rows on the finite strip, grid points per cell


def shares(kx, R=R_HW):
    """Lossless modes of the matched strip and their energy shares.
       Returns rows (x, sign U, Lam_P, Lam_N) with Lam = group energy / |U|."""
    A, B, K, wC = fd_pencil(R, 1.0, 1.0, kx, M, MGRID)
    h = 1.0/MGRID; host = h*np.ones(len(wC)); rowC = wC - host
    lam, V = eig(A, B)
    out = []
    for i in range(len(lam)):
        x = lam[i].real; u = V[:, i].real
        if not np.isfinite(x) or x <= 0:
            continue
        Eh = x*np.sum(host*u*u); Mh = u @ K @ u + kx**2*np.sum(host*u*u)
        Er = x*rowC*u*u; Mr = kx**2*rowC*u*u
        p, q = rowC > 0, rowC < 0
        U = Eh + Mh + Er.sum() + Mr.sum()
        out.append((x, np.sign(U), (Eh + Mh + Er[p].sum() + Mr[p].sum())/abs(U),
                    -(Er[q].sum() + Mr[q].sum())/abs(U)))
    return np.array(out)


def lossy_omegas(kx, d_host, d_pos, d_nic, R=R_HW):
    """Complex mode frequencies omega (units v_p/a) of the lossy finite strip."""
    A, B, K, wC = fd_pencil(R, 1.0, 1.0, kx, M, MGRID)
    h = 1.0/MGRID; n = len(wC); rowC = wC - h
    d = np.where(rowC > 0, d_pos, np.where(rowC < 0, d_nic, 0.0))
    Ad = K*(1 - 1j*d_host) + kx**2*np.diag(h*(1 - 1j*d_host)*np.ones(n) + rowC - 1j*np.abs(rowC)*d)
    Bd = np.diag(h*(1 + 1j*d_host)*np.ones(n) + rowC + 1j*np.abs(rowC)*d)
    lam = eig(Ad, Bd, right=False)
    lam = lam[np.isfinite(lam)]
    w = np.sqrt(lam.astype(complex))
    return np.where(w.real < 0, -w, w)


def GHz(w):
    return V_P*w/A_CELL/2/np.pi/1e9


if __name__ == "__main__":
    print("=== First-order participation ratios, matched 10-pF design ===")
    for order in range(1, 7):
        kx = order*np.pi/W_A
        s = shares(kx); ng, ps = s[s[:, 1] < 0], s[s[:, 1] > 0]
        Lam = ng[:, 2] + ng[:, 3]
        lo = (ng[:, 2]/ng[:, 3]).max(); hi = (ps[:, 2]/ps[:, 3]).min()
        print(f"order m={order} (kx a={kx:.3f}): {len(ng)} negative-energy modes, "
              f"{GHz(np.sqrt(ng[:,0].min())):.3f}-{GHz(np.sqrt(ng[:,0].max())):.3f} GHz, "
              f"Lambda = {Lam.min():.1f}-{Lam.max():.1f};  NIC gain window {lo:.3f} < rho < {hi:.3f}")

    kx = np.pi/W_A
    print("\n=== Direct lossy solve, order m=1, uniform Q = 100 (d = 0.01) ===")
    w = lossy_omegas(kx, 0.01, 0.01, 0.01); i = np.argmax(w.imag)
    print(f"max growth sigma = {V_P*w.imag[i]/A_CELL:.2e} 1/s at {GHz(w.real[i]):.3f} GHz "
          f"(sigma/omega = {w.imag[i]/w.real[i]:.4f}; e-folding {A_CELL/V_P/w.imag[i]*1e9:.1f} ns)")

    print("\n=== NIC gain d_NIC = -rho * d, host and positive rows d = 0.01 ===")
    for rho in [0.0, 0.5, 0.7, 0.8, 1.0, 1.3, 1.35, 1.4, 1.6]:
        w = lossy_omegas(kx, 0.01, 0.01, -rho*0.01); i = np.argmax(w.imag)
        tag = "stable" if w.imag[i] <= 1e-9 else "unstable"
        print(f"rho={rho:.2f}: max Im(omega)/Re(omega) = {w.imag[i]/w.real[i]:+.4f} at {GHz(w.real[i]):.3f} GHz  [{tag}]")
