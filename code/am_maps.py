"""
am_maps.py -- stability maps behind Fig. 4(a,b). Exact transfer matrix (am_model.classify).
  am_infinite_map.npz : (gamma, kx a) at R = 3, uniform matching gamma_+ = gamma_- = gamma   [~3 min]
  am_tolerance_map.npz: (gamma_+, gamma_-) at the design point R = 3.003, kx a = 0.6974      [~5 min]
code: 0 stable, 1 zero-frequency growth, 2 oscillatory growth.
Usage: python am_maps.py [--coarse]   (--coarse: 4x fewer points, ~1 min, for a quick rerun)
"""
import sys, time, numpy as np
from am_model import classify
coarse = "--coarse" in sys.argv
s = 2 if coarse else 1

t = time.time()
R = 3.0
G = np.linspace(0.0, 1.5, 121)[::s]; KX = np.linspace(0.01, 2.0, 200)[::s]
code = np.zeros((len(G), len(KX)), int)
for i, g in enumerate(G):
    for j, kx in enumerate(KX):
        zf, osc = classify(R, g, g, kx, n=30001)
        code[i, j] = 2 if osc else (1 if zf else 0)
np.savez("am_infinite_map.npz", G=G, KX=KX, code=code, R=R)
print(f"am_infinite_map.npz  {code.shape}  {time.time()-t:.0f}s")

t = time.time()
R, k = 3.003, 0.6974
GP = np.linspace(0.6, 1.4, 161)[::s]; GN = np.linspace(0.6, 1.4, 161)[::s]
code = np.zeros((len(GP), len(GN)), int)
for i, gp in enumerate(GP):
    for j, gn in enumerate(GN):
        zf, osc = classify(R, gp, gn, k, n=30001)
        code[i, j] = 2 if osc else (1 if zf else 0)
np.savez("am_tolerance_map.npz", GP=GP, GN=GN, code=code, R=R, k=k)
print(f"am_tolerance_map.npz {code.shape}  {time.time()-t:.0f}s")
