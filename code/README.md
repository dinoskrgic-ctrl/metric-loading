# Verification code for "Metric Loading" (metric_loading.tex)

```
python am_checks.py      # 65 PASS/FAIL checks of every number in the paper   (~5 min)
python am_maps.py        # stability maps for Fig. 4(a,b)                      (~8 min; --coarse ~2.5 min)
python am_loss.py        # loss growth rates and the NIC gain window (Sec. VI) (~10 min)
python am_figs.py        # Figs. 2, 3, 4 from the maps                         (~1 min)
python am_figs.py --no-latex   # same, if your LaTeX lacks cm-super (mathtext fonts)
```

Requires numpy, scipy, matplotlib and scienceplots (the last is only needed for the figures).

## Files

| file | role |
|---|---|
| `am_model.py` | All the physics: closed forms, the exact transfer matrix, the finite-difference strip solver, hardware constants. Every other script imports from here. |
| `am_checks.py` | The verification suite. Each check names the paper equation or section it supports and which solver it uses. |
| `am_maps.py` | Builds `am_infinite_map.npz` (Fig. 4a) and `am_tolerance_map.npz` (Fig. 4b). |
| `am_loss.py` | Loss (Sec. VI): energy participation ratios, first-order growth rates, the NIC gain window, and the direct complex eigen-solve of the lossy strip. |
| `am_figs.py` | Builds `fig_krein_band.pdf`, `fig_phase.pdf` and `fig_matching.pdf`. |
| `*.npz` | The stored maps used for the published figures. |

## Units and model

Lengths are in units of the row spacing a, velocities in units of v_p, x = (ω/v_p)², and R = r₀/a with r₀ = C_L/2C′. A single equation covers every case in the paper (Eq. 19):

    -φ'' + k_x² W_γ φ = x W φ,    W = 1 + 2Σ r_n δ(y−n),    W_γ = 1 + 2Σ γ_n r_n δ(y−n)

- γ = 1 is matched (covariant) loading.
- γ = 0 is ordinary capacitors with no series elements.
- γ₊ and γ₋ are the separate matching fractions on the positive and negative rows.

## The two independent solvers

The verification rests on agreement between two methods that share no code beyond the parameter values.

**[TM] Exact transfer matrix.** Implemented in `z_general`, this is the half-trace of the doubled cell for arbitrary γ₊ and γ₋ (Appendix B):

    z = cosh 2μ + (g₁+g₂) sinh 2μ/(2μ) + g₁g₂ sinh²μ/(2μ²),    μ² = k_x² − x,    g = ±2R(γk_x² − x)

`classify` decides the stability of the **infinite** strip from z(x):
- *Zero-frequency growth:* some x < 0 has |z| ≤ 1. This is a real Bloch mode with ω² < 0.
- *Oscillatory growth:* z(x) has a turning point with |z| < 1 at real x. Near such a point, real Bloch wavenumbers require complex x, which means a complex pole pair.

Each turning point is refined with a bounded optimizer. Without this, the exact Bragg touchings (z = 1) are sampled slightly below 1 on a grid and misread as instabilities. That bug occurred in an earlier version and was fixed.

**[FD] Finite-difference strip.** Implemented in `fd_pencil` and `fd_spectrum`, this is a direct discretization of the quadratic forms on a **finite** Dirichlet strip. It solves the generalized eigenproblem A u = x B u with a dense general eigensolver, so it does not assume real eigenvalues. Instability is read off as any eigenvalue with Re x < 0 or Im x ≠ 0.

For the matched case, `fd_kappa2_negative` uses the symmetric form K^(−1/2) W K^(−1/2), which is exactly the argument of Appendix A. It is used to count the negative-energy modes.

`single_row_pencil` is the same FD solver with one isolated row. It tests the isolated-row formulas (Eq. 20 and the over-match onset).

## What `am_checks.py` checks, section by section

1. **Threshold equation (Eqs. 13, 16, 17).** Both asymptotic limits; that χ* decreases monotonically with r₀; and that the band edges sit at z = ±1.
2. **Negative-energy modes (Secs. II–III, App. A).**
   - Number of κ² < 0 modes = number of negative rows (10 of 10) at R = 0.15 and R = 3.
   - All such modes lie inside [1/R, χ*].
   - A κ² < 0 eigenvector has a negative capacitance-weighted norm (Eq. 8).
3. **Matched threshold (Eqs. 14–15).**
   - The strip is unstable just below χ* and stable just above, with no complex modes at any k_x.
   - σ_max = √(χ*² − k_x²) agrees to 10⁻⁵ on the infinite strip.
   - Finite strips approach σ_max from below (M = 10, 20, 40 rows), so the infinite-strip criterion is conservative.
4. **Ordinary capacitors, γ = 0 (Sec. IV).**
   - Unstable at every k_x, with exactly one growing mode per negative row (Sylvester count).
   - The isolated-row growth formula (Eq. 20) agrees with FD to 0.06%.
   - Growth increases as the strip narrows.
5. **Under-matching (Eq. 21).**
   - For γ = 0.75, 0.5, 0.25 the threshold is χ*(γr₀) on both solvers, with no oscillatory modes.
   - z(x) is monotone along x < 0 at threshold (Appendix B).
6. **Over-matching (Eq. 23).**
   - For an isolated row, the complex pair appears between 0.95 and 1.05 of k_x,c = 1/(2r₀√(γ−1)).
   - On the lattice, the onset matches γ_c = 1 + 1/(4R²k_x²) to 10⁻¹⁰ at four values of k_x. This is the claim that it does not depend on the lattice.
   - The two colliding modes carry opposite energy signs.
7. **Separate row matching (Sec. IV).**
   - γ₊ = 0.95 oscillates on both solvers.
   - At nominal matching, independent errors of ±0.5% are stable and ±1% are not.
   - The biased design (1.2, 0.95) is stable over the whole box of half-widths 0.15 in γ₊ and 0.10 in γ₋ [TM], and at its corners and centre [FD].
   - In terms of the inductors themselves (L ∝ 1/γ), independent errors of ±12.5% on the passive rows and ±10% on the NIC rows are stable, while ±15%/±10% is not [TM].
8. **Hardware (Table II, Sec. V).** Every number in the table and in the design list:
   - the margin, the cutoff and the 0.64–1.17 GHz band;
   - L_L, the discrete-row correction, the 7.92 pF threshold-test value and the 0.79–1.057 window;
   - the 1.29 GHz oscillation onset, the open-edge growth rate and the NIC count.

`am_maps.py --coarse` reproduces every other point of both stored maps exactly (12,661 of 12,661).

9. **Loss and NIC gain (Sec. VI).**
   - Participation ratio Λ = 3.6–7.8 for the band modes of the lowest transverse order.
   - First-order NIC gain window 0.77 < ρ < 1.37.
   - Direct lossy solve at Q = 100: largest growth 4.6×10⁸ s⁻¹ near 1.14 GHz; ρ = 0.7 unstable, 0.8 and 1.3 stable, 1.4 unstable.

## Known limitations of the checks

- **FD discretization error.** At R = 0.15 the FD modes sit at χ = 6.6815 against the exact 6.6667, an error of 0.2% that comes from the grid. The band check uses a 0.3% tolerance for this reason.
- **Finite-size effects.** FD strips have finite length, so they are slightly more stable than the infinite strip near every boundary. Example: γ₊ = 0.97 is stable at M = 24 but unstable on the infinite strip. All numbers quoted in the paper are infinite-strip [TM] values, which are the conservative ones.
- **What is not proven in closed form.**
  - The positive-row tolerance (γ₊ ≳ 0.98 at nominal matching) and the biased tolerance box are numerical maps.
  - Monotonicity of z along x < 0 for γ ≤ 1 was checked numerically, not proven.
- **Ideal elements** in Secs. II–V; Sec. VI adds frequency-independent loss tangents (constant Q) to first order and in a direct solve. No element dispersion, no lumped-circuit (SPICE) model of the NICs.
- **Both solvers solve the same continuum model**, so agreement tests the closed forms, not the model.
