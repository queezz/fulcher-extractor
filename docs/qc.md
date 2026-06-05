# QC

QC should make failed or suspicious fits visible before the intensity tables are
used downstream.

Initial QC outputs should include:

- Per-line fit status.
- Residual plots for each line or blend group.
- Integrated intensity, uncertainty, and signal-to-noise.
- Fitted center offset from database wavelength.
- Width and baseline sanity checks.
- Batch-level summary CSV.

