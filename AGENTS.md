# AGENTS.md

This repo is the Fulcher-alpha line extraction package.

## Boundaries

- Consume calibrated SpectroCube files.
- Keep instrument calibration and SpectroCube generation upstream in
  `echelle_spectra` and the Echelle workflow.
- Keep Boltzmann/coronal physics analysis downstream in `fulcheranalyzer`.
- Keep experiment-specific paths in workflow repos or ignored `local/` files.

## Expected Implementation Surface

- `spectrocube_io.py`: read cube files and select wavelength windows.
- `line_database.py`: load Fulcher-alpha transition metadata.
- `line_models.py`: Gaussian, blended Gaussian, and baseline models.
- `fit.py`: fitting routines and uncertainty estimates.
- `extract.py`: high-level extraction workflow.
- `qc.py`: fit diagnostics, warning flags, and plots.
- `output.py`: `fulcheranalyzer`-compatible CSV output.

## Working Rules

- Preserve metadata needed to trace every intensity back to cube, shot, frame,
  wavelength window, and fit model.
- Prefer explicit TOML/CSV resources for line lists and blend groups.
- Add synthetic tests before fitting real data in bulk.
- Do not commit local datasets or machine-specific absolute paths.

