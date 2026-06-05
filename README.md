# 2026 Fulcher Extractor

SpectroCube consumer for extracting Fulcher-alpha line intensities from
calibrated spectra.

## Purpose

This repo owns the reusable extraction layer between Echelle SpectroCube files
and `fulcheranalyzer` intensity tables.

It should contain:

- Fulcher-alpha wavelength and transition metadata.
- Blend-group definitions.
- Gaussian and blended-Gaussian fitting models.
- Batch extraction logic for SpectroCube files.
- Fit QC summaries and diagnostic plots.
- Writers for `fulcheranalyzer`-compatible intensity and error CSVs.

It should not contain:

- Raw detector-frame calibration logic.
- Experiment-specific absolute local paths.
- Boltzmann or coronal-model analysis code.

## Downstream Flow

```text
SpectroCube .nc
  -> fulcher_extractor
  -> intensity/error CSVs + QC
  -> fulcheranalyzer
  -> Boltzmann/coronal analysis
```

## Development

```bash
python -m pip install -e ".[dev,docs]"
pytest
mkdocs serve
```

