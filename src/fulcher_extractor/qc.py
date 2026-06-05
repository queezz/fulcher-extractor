"""Fit quality-control summaries and diagnostic plot helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .line_database import FulcherLine
from .spectrocube_io import Spectrum


def plot_region(
    spectrum: Spectrum,
    *,
    wavelength_min_nm: float = 600.0,
    wavelength_max_nm: float = 630.0,
    lines: list[FulcherLine] | None = None,
    output_path: str | Path | None = None,
):
    """Plot a wider wavelength region for background inspection."""
    window = spectrum.window(wavelength_min_nm, wavelength_max_nm)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(window.wavelength_nm, window.intensity, lw=0.8, color="black")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(f"Intensity [{spectrum.intensity_units}]")
    title = f"{spectrum.source_path.name}"
    if spectrum.selectors:
        title += " " + ", ".join(f"{k}={v}" for k, v in spectrum.selectors.items())
    ax.set_title(title)
    if lines:
        ymax = ax.get_ylim()[1]
        for line in lines:
            if wavelength_min_nm <= line.wavelength_nm <= wavelength_max_nm:
                ax.axvline(line.wavelength_nm, color="tab:red", lw=0.6, alpha=0.45)
                ax.text(
                    line.wavelength_nm,
                    ymax,
                    f"Q{line.N} {line.band}",
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=7,
                    color="tab:red",
                )
    fig.tight_layout()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    return fig
