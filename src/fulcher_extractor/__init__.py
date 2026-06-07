"""Fulcher-alpha line extraction from calibrated SpectroCube spectra."""

from .fit import (
    ContaminantSpec,
    FitConfig,
    InstrumentWidthEstimate,
    LineFitResult,
    estimate_instrument_width,
    fit_line_group,
    fit_line_with_contaminants,
    fit_single_line,
    group_close_lines,
)
from .line_database import FulcherLine, filter_lines, load_lines
from .line_policy import LinePolicy, apply_line_policies, load_line_policies
from .spectrocube_io import Spectrum, load_spectrum

__all__ = [
    "FitConfig",
    "ContaminantSpec",
    "FulcherLine",
    "InstrumentWidthEstimate",
    "LineFitResult",
    "LinePolicy",
    "Spectrum",
    "apply_line_policies",
    "estimate_instrument_width",
    "filter_lines",
    "fit_line_group",
    "fit_line_with_contaminants",
    "fit_single_line",
    "group_close_lines",
    "load_lines",
    "load_line_policies",
    "load_spectrum",
]
