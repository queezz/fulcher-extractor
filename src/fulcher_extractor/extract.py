"""High-level Fulcher-alpha extraction workflow."""

from __future__ import annotations

import numpy as np

from .fit import FitConfig, LineFitResult, fit_line_group, group_close_lines
from .line_database import FulcherLine, filter_lines, load_lines
from .spectrocube_io import Spectrum


def extract_lines(
    spectrum: Spectrum,
    *,
    lines: list[FulcherLine] | None = None,
    isotopologue: str = "H2",
    wavelength_min_nm: float | None = 600.0,
    wavelength_max_nm: float | None = 630.0,
    config: FitConfig | None = None,
) -> list[LineFitResult]:
    """Fit selected Fulcher-alpha lines from one spectrum."""
    all_lines = lines if lines is not None else load_lines()
    selected = filter_lines(
        all_lines,
        isotopologue=isotopologue,
        wavelength_min_nm=wavelength_min_nm,
        wavelength_max_nm=wavelength_max_nm,
    )
    cfg = config or FitConfig()
    if cfg.close_neighbor_threshold_nm is None:
        threshold = (
            2.0 * np.sqrt(2.0 * np.log(2.0)) * cfg.instrument_sigma_nm
            if cfg.instrument_sigma_nm is not None
            else 0.07
        )
    else:
        threshold = cfg.close_neighbor_threshold_nm
    results: list[LineFitResult] = []
    for group in group_close_lines(selected, threshold):
        results.extend(fit_line_group(spectrum, group, config=cfg))
    return results
