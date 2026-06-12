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

For batch SpectroCube runs, keep extraction and physics analysis as separate
steps:

```bash
fulcher-h2-dataset --plan h2_dataset_plan.toml extract
fulcher-h2-dataset --plan h2_dataset_plan.toml plot
fulcher-analyze-batch --plan h2_dataset_plan.toml
```

The plan should include an `[analyze]` section for downstream analyzer paths.
The optional `plot` step regenerates QC plots from existing extraction fit
reports without refitting spectra. By default, line-fit QC is written as one
multi-page PDF per shot.

Extraction writes `extraction_progress.jsonl` as frames complete. Use
`fulcher-h2-dataset --plan h2_dataset_plan.toml extract --resume` to skip
checkpointed frames whose intensity, error, and fit-report artifacts are
already present.

## Development

Use the Fulcher virtual environment explicitly. Future agents should run tests
and scripts through this interpreter:

```bash
~/.venvs/fulcher/bin/python -m pip install -e ".[dev,docs]"
~/.venvs/fulcher/bin/python -m pytest
~/.venvs/fulcher/bin/python -m mkdocs serve
```

Ignored local run outputs should use a date-time prefix:

```text
local/runs/YYYYMMDD-HHMM-short-description/
```
