# Metric loading: An exactly solvable stability threshold for non-Foster transmission-line metamaterials

Dino Škrgić, Dado Škrgić, Dejan Stojkovic

Manuscript (REVTeX 4.2) and the code that reproduces every number and figure in it.

## Layout

```
paper/   metric_loading.tex        manuscript
         references.bib            bibliography (cited entries only)
         fig_krein_band.pdf        Fig. 2
         fig_phase.pdf             Fig. 3
         fig_matching.pdf          Fig. 4   (Fig. 1 is TikZ/circuitikz inside the .tex)
code/    am_model.py               closed forms, exact transfer matrix, finite-difference solver
         am_checks.py              65 PASS/FAIL checks of the numbers in the paper
         am_loss.py                loss growth rates and the NIC gain window (Sec. VI)
         am_maps.py                stability maps behind Fig. 4(a,b) -> *.npz
         am_figs.py                Figs. 2-4 from the maps
         am_*_map.npz              stored maps used for the figures
         checks_output.txt         output of the last full check run
         am_loss_output.txt        output of the last am_loss.py run
         README.md                 what each check verifies, and the known limitations
```

## Build the paper

```
cd paper
pdflatex metric_loading
bibtex   metric_loading
pdflatex metric_loading
pdflatex metric_loading
```

Requires REVTeX 4.2, `tikz` and `circuitikz`.

## Reproduce the numbers and figures

```
cd code
pip install -r ../requirements.txt
python am_checks.py            # ~5 min, expect "65/65 checks passed."
python am_loss.py              # ~10 min
python am_maps.py              # ~12 min (--coarse: ~3 min)
python am_figs.py              # writes fig_*.pdf; add --no-latex if LaTeX lacks cm-super
cp fig_*.pdf ../paper/
```
