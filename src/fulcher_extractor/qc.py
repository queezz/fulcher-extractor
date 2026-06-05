"""Fit quality-control summaries and diagnostic plot helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .fit import LineFitResult
from .line_database import FulcherLine
from .line_models import gaussian_area_model
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


def plot_line_fit(
    spectrum: Spectrum,
    result: LineFitResult,
    *,
    components: list[dict[str, float | str]] | None = None,
    neighbor_lines: list[FulcherLine] | None = None,
    output_path: str | Path | None = None,
):
    """Plot a zoomed line fit with data, Gaussian component(s), sum, and residual."""
    window = spectrum.window(result.window_min_nm, result.window_max_nm)
    x = window.wavelength_nm
    y = window.intensity
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    component_specs = components
    if component_specs is None and result.success:
        component_specs = [
            {
                "label": result.line_id,
                "amplitude": result.amplitude,
                "center_nm": result.center_nm,
                "sigma_nm": result.sigma_nm,
            }
        ]
    component_specs = component_specs or []

    x_model = np.linspace(result.window_min_nm, result.window_max_nm, 500)
    component_profiles = []
    for spec in component_specs:
        profile = gaussian_area_model(
            x_model,
            float(spec["amplitude"]),
            float(spec["center_nm"]),
            float(spec["sigma_nm"]),
        )
        component_profiles.append((str(spec.get("label", "component")), profile))

    if component_profiles:
        sum_profile = np.sum([profile for _, profile in component_profiles], axis=0)
        y_sum_raw = result.baseline_offset + sum_profile
        y_fit_at_data = result.baseline_offset + np.sum(
            [
                gaussian_area_model(
                    x,
                    float(spec["amplitude"]),
                    float(spec["center_nm"]),
                    float(spec["sigma_nm"]),
                )
                for spec in component_specs
            ],
            axis=0,
        )
        residual = y - y_fit_at_data
    else:
        y_sum_raw = np.full_like(x_model, np.nan)
        residual = np.full_like(y, np.nan)

    fig, (ax, ax_resid) = plt.subplots(
        2,
        1,
        figsize=(7.5, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax.plot(x, y, ".", ms=3, color="black", label="data")
    ax.axhline(
        result.baseline_offset,
        color="0.45",
        lw=0.9,
        ls=":",
        label=f"baseline {result.baseline_offset:.3g}",
    )
    for label, profile in component_profiles:
        ax.plot(
            x_model,
            result.baseline_offset + profile,
            lw=1.0,
            ls="--",
            alpha=0.75,
            label=label,
        )
    if component_profiles:
        ax.plot(x_model, y_sum_raw, lw=1.8, color="tab:red", label="fit sum")
    ax.axvline(
        result.rest_wavelength_nm,
        color="tab:blue",
        lw=0.9,
        alpha=0.65,
        label="database",
    )
    if np.isfinite(result.expected_center_nm):
        ax.axvline(
            result.expected_center_nm,
            color="tab:purple",
            lw=0.9,
            alpha=0.65,
            ls=":",
            label="expected",
        )
    if np.isfinite(result.center_nm):
        ax.axvline(
            result.center_nm,
            color="tab:red",
            lw=0.9,
            alpha=0.65,
            ls="-.",
            label="fit center",
        )
    if neighbor_lines:
        _plot_neighbor_lines(ax, result, neighbor_lines)
    ax.set_ylabel(f"Intensity [{spectrum.intensity_units}]")
    ax.set_title(
        f"{spectrum.shot_id or spectrum.source_path.stem} "
        f"{_selector_label(spectrum)} {result.line_id} {result.status}"
    )
    ax.legend(loc="best", fontsize=8)

    ax_resid.axhline(0.0, color="0.45", lw=0.8)
    ax_resid.plot(x, residual, ".", ms=3, color="tab:gray")
    ax_resid.set_xlabel("Wavelength [nm]")
    ax_resid.set_ylabel("Residual")
    ax_resid.text(
        0.01,
        0.95,
        (
            f"area={result.amplitude:.4g}, center-rest="
            f"{result.center_offset_from_rest_nm:.4g} nm, rms={result.residual_rms:.3g}"
        ),
        transform=ax_resid.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )
    fig.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    return fig


def _plot_neighbor_lines(
    ax,
    result: LineFitResult,
    neighbor_lines: list[FulcherLine],
) -> None:
    colors = {"0-0": "tab:green", "1-1": "tab:orange", "2-2": "tab:cyan", "3-3": "tab:pink"}
    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.04 * (ymax - ymin)
    for line in neighbor_lines:
        if line.line_id == result.line_id:
            continue
        if not (result.window_min_nm <= line.wavelength_nm <= result.window_max_nm):
            continue
        color = colors.get(line.band, "0.5")
        ax.axvline(line.wavelength_nm, color=color, lw=0.7, alpha=0.45)
        ax.text(
            line.wavelength_nm,
            y_text,
            f"Q{line.N} {line.band}",
            rotation=90,
            va="top",
            ha="center",
            fontsize=7,
            color=color,
        )


def _selector_label(spectrum: Spectrum) -> str:
    if not spectrum.selectors:
        return ""
    return " ".join(f"{key}={value}" for key, value in spectrum.selectors.items())
