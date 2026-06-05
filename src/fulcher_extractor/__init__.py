"""Fulcher-alpha line extraction from calibrated SpectroCube spectra."""

from .fit import (
    FitConfig,
    InstrumentWidthEstimate,
    LineFitResult,
    estimate_instrument_width,
    fit_line_group,
    fit_single_line,
    group_close_lines,
)
from .line_database import FulcherLine, filter_lines, load_lines
from .spectrocube_io import Spectrum, load_spectrum

__all__ = [
    "FitConfig",
    "FulcherLine",
    "InstrumentWidthEstimate",
    "LineFitResult",
    "Spectrum",
    "estimate_instrument_width",
    "filter_lines",
    "fit_line_group",
    "fit_single_line",
    "group_close_lines",
    "load_lines",
    "load_spectrum",
]
