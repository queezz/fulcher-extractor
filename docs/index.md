# Fulcher Extractor

`fulcher-extractor` turns calibrated SpectroCube spectra into fitted
Fulcher-alpha line intensities, uncertainty estimates, and QC artifacts.

The package is intentionally separate from `fulcheranalyzer`: extraction is a
spectral fitting problem, while `fulcheranalyzer` owns the Boltzmann and coronal
analysis after intensities are measured.

