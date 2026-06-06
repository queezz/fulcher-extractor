"""Shared plotting style for Fulcher QC figures."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib.pyplot as plt

BAND_COLORS = {
    "0-0": "#d55e00",
    "1-1": "#e69f00",
    "2-2": "#0072b2",
    "3-3": "#cc79a7",
}

COMPONENT_COLORS = {
    "0-0": "#d55e00",
    "1-1": "#009e73",
    "2-2": "#0072b2",
    "3-3": "#cc79a7",
}

FIT_SUM_COLOR = "#b2182b"
BASELINE_COLOR = "0.55"
MARKER_COLOR = "0.28"

QC_RCPARAMS = {
    "axes.grid": False,
    "axes.linewidth": 0.8,
    "font.family": "serif",
    "font.size": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
}


@contextmanager
def fulcher_qc_style() -> Iterator[None]:
    """Apply archive-inspired matplotlib defaults for one QC figure."""
    with plt.rc_context(QC_RCPARAMS):
        yield


def band_color(band: str, *, component: bool = False) -> str:
    """Return a stable color for a diagonal Fulcher band."""
    palette = COMPONENT_COLORS if component else BAND_COLORS
    return palette.get(band, "0.4")


def compact_line_label(line_id: str) -> str:
    """Convert H2_Q9_1-1 to H2 Q9(1-1) for direct plot labels."""
    parts = line_id.split("_")
    if len(parts) >= 3:
        return f"{parts[0]} {parts[1]}({parts[2]})"
    return line_id.replace("_", " ")


def set_inward_ticks(ax) -> None:
    """Use inward ticks on all axes, matching the notebook figures."""
    ax.tick_params(which="both", direction="in", top=False, right=False)
    ax.minorticks_on()
