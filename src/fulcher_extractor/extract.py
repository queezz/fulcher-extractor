"""High-level Fulcher-alpha extraction workflow."""

from __future__ import annotations

from .fit import FitConfig, LineFitResult, fit_single_line
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
    return [fit_single_line(spectrum, line, config=config) for line in selected]
