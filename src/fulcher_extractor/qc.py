"""Fit quality-control summaries and diagnostic plot helpers."""

from __future__ import annotations

from pathlib import Path
import math

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from .fit import LineFitResult
from .line_database import FulcherLine
from .line_models import gaussian_area_model
from .plot_style import (
    BASELINE_COLOR,
    FIT_SUM_COLOR,
    band_color,
    compact_line_label,
    fulcher_qc_style,
    set_inward_ticks,
)
from .spectrocube_io import Spectrum

LINE_COLORS = {
    "0-0": band_color("0-0", component=True),
    "1-1": band_color("1-1", component=True),
    "2-2": band_color("2-2", component=True),
    "3-3": band_color("3-3", component=True),
}

FitComponentSpec = dict[str, float | str]


def plot_region(
    spectrum: Spectrum,
    *,
    wavelength_min_nm: float = 600.0,
    wavelength_max_nm: float = 630.0,
    lines: list[FulcherLine] | None = None,
    label_lines: list[FulcherLine] | None = None,
    guide_lines: list[FulcherLine] | None = None,
    show_guides: bool = True,
    show_all_line_labels: bool = False,
    labels_above_axes: bool = True,
    output_path: str | Path | None = None,
):
    """Plot a wider wavelength region for background inspection."""
    window = spectrum.window(wavelength_min_nm, wavelength_max_nm)
    with fulcher_qc_style():
        fig, ax = plt.subplots(figsize=(11, 4.4))
        ax.plot(window.wavelength_nm, window.intensity, lw=0.75, color="black")
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel(f"Intensity [{spectrum.intensity_units}]")
        set_inward_ticks(ax)
        title = _spectrum_label(spectrum)
        ax.text(
            0.985,
            1.02,
            title,
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=9,
            clip_on=False,
        )
        if lines:
            labelled = lines if show_all_line_labels else label_lines
            labelled = labelled if labelled is not None else lines
            guides = guide_lines if guide_lines is not None else labelled
            if show_guides:
                _plot_region_guides(ax, guides, wavelength_min_nm, wavelength_max_nm)
            _plot_band_rails(
                ax,
                lines,
                wavelength_min_nm,
                wavelength_max_nm,
                label_lines=labelled,
                show_all_line_labels=show_all_line_labels,
                labels_above_axes=labels_above_axes,
            )
        fig.tight_layout()
        if labels_above_axes and lines:
            fig.subplots_adjust(top=0.66)
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
    show_database_markers: bool = True,
    show_fitted_markers: bool = True,
    summary_location: str = "residual",
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
                "rest_wavelength_nm": result.rest_wavelength_nm,
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
        component_profiles.append((spec, str(spec.get("label", "component")), profile))

    if component_profiles:
        sum_profile = np.sum([profile for _, _, profile in component_profiles], axis=0)
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

    with fulcher_qc_style():
        fig, (ax, ax_resid) = plt.subplots(
            2,
            1,
            figsize=(7.5, 5.2),
            sharex=True,
            gridspec_kw={"height_ratios": [3.2, 1]},
        )
        ax.plot(
            x,
            y,
            ":+",
            lw=1.0,
            ms=4.5,
            mew=0.85,
            color="black",
            label="data",
        )
        ax.axhline(
            result.baseline_offset,
            color=BASELINE_COLOR,
            lw=0.8,
            ls=":",
            label="baseline",
        )
        for spec, label, profile in component_profiles:
            color = _component_color(spec, label)
            is_target = label == result.line_id
            is_contaminant = _is_contaminant_component(spec)
            ls = "-" if is_target else "--"
            ax.plot(
                x_model,
                result.baseline_offset + profile,
                lw=1.25 if is_target else 1.0,
                ls=ls,
                alpha=0.9 if (is_target or is_contaminant) else 0.72,
                color=color,
                label=_component_display_label(spec, label),
            )
            if _should_hatch_component():
                ax.fill_between(
                    x_model,
                    result.baseline_offset,
                    result.baseline_offset + profile,
                    facecolor="none",
                    edgecolor=color,
                    hatch="////",
                    linewidth=0.0,
                    alpha=0.26,
                )
            _label_component(ax, x_model, result.baseline_offset + profile, spec, label, color)
        if component_profiles:
            ax.plot(x_model, y_sum_raw, lw=1.8, color=FIT_SUM_COLOR, label="fit sum")
        if show_database_markers or show_fitted_markers:
            _plot_component_markers(
                ax,
                result,
                component_specs,
                show_database=show_database_markers,
                show_fitted=show_fitted_markers,
            )
        if neighbor_lines:
            _plot_neighbor_lines(ax, result, neighbor_lines)
        ax.set_ylabel(f"Intensity [{spectrum.intensity_units}]")
        ax.set_title(f"{_fit_title(result, component_specs)} fit QC", loc="left", fontsize=11)
        ax.set_title(_spectrum_label(spectrum), loc="right", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, frameon=True, handlelength=1.8)
        set_inward_ticks(ax)

        ax_resid.axhline(0.0, color="0.45", lw=0.8)
        ax_resid.plot(x, residual, "+", ms=3.8, mew=0.7, color="0.35")
        ax_resid.set_xlabel("Wavelength [nm]")
        ax_resid.set_ylabel("Residual")
        _plot_fit_note(ax_resid if summary_location == "residual" else ax, result)
        if summary_location == "residual":
            ax_resid.set_title(f"status: {result.status}", loc="left", fontsize=7, pad=2)
        set_inward_ticks(ax_resid)
        fig.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    return fig


def plot_line_fit_page(
    spectrum: Spectrum,
    results: list[LineFitResult],
    *,
    columns: int = 5,
    output_path: str | Path | None = None,
):
    """Plot all fitted line groups for one spectrum on one large QC page."""
    groups = _line_fit_groups(results)
    if not groups:
        raise ValueError("At least one LineFitResult is required.")
    if columns < 1:
        raise ValueError("columns must be >= 1.")

    rows = math.ceil(len(groups) / columns)
    width = max(11.0, columns * 3.0)
    height = max(8.5, rows * 2.25)
    with fulcher_qc_style():
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(width, height),
            squeeze=False,
            sharey=False,
        )
        for ax in axes.flat:
            ax.set_visible(False)

        for ax, group in zip(axes.flat, groups):
            ax.set_visible(True)
            _plot_line_fit_panel(spectrum, group.reference, group.components, ax=ax)

        fig.suptitle(f"{_spectrum_label(spectrum)} line-fit QC", fontsize=14, y=0.995)
        fig.supxlabel("Wavelength [nm]", fontsize=11)
        fig.supylabel(f"Intensity [{spectrum.intensity_units}]", fontsize=11)
        fig.tight_layout(rect=(0.025, 0.025, 0.995, 0.975))

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".pdf":
            with PdfPages(output) as pdf:
                pdf.savefig(fig)
        else:
            fig.savefig(output, dpi=180)
    return fig


def write_line_fit_qc(
    spectrum: Spectrum,
    results: list[LineFitResult],
    *,
    pdf_path: str | Path | None = None,
    individual_dir: str | Path | None = None,
    save_individual_pngs: bool = False,
    columns: int = 5,
) -> list[Path]:
    """Write line-fit QC artifacts for one spectrum.

    The PDF page is intended for normal human review. Individual PNGs remain
    available for detailed fitting checks, but are opt-in for batch runs.
    """
    written: list[Path] = []
    groups = _line_fit_groups(results)

    if pdf_path is not None:
        pdf_output = Path(pdf_path)
        plot_line_fit_page(spectrum, results, columns=columns, output_path=pdf_output)
        written.append(pdf_output)

    if save_individual_pngs:
        if individual_dir is None:
            raise ValueError("individual_dir is required when save_individual_pngs=True.")
        png_dir = Path(individual_dir)
        png_dir.mkdir(parents=True, exist_ok=True)
        for group in groups:
            output = png_dir / f"{_safe_line_group_name(group.reference)}.png"
            fig = plot_line_fit(
                spectrum,
                group.reference,
                components=group.components,
                output_path=output,
            )
            plt.close(fig)
            written.append(output)

    return written


class _LineFitGroup:
    def __init__(
        self,
        reference: LineFitResult,
        components: list[FitComponentSpec],
    ) -> None:
        self.reference = reference
        self.components = components


def _line_fit_groups(results: list[LineFitResult]) -> list[_LineFitGroup]:
    by_group: dict[str, list[LineFitResult]] = {}
    for result in results:
        key = result.blend_group_id or result.line_id
        by_group.setdefault(key, []).append(result)

    groups = []
    for grouped in by_group.values():
        ordered = sorted(grouped, key=lambda item: (item.rest_wavelength_nm, item.line_id))
        reference = ordered[0]
        groups.append(
            _LineFitGroup(
                reference=reference,
                components=[_component_from_result(result) for result in ordered],
            )
        )
    return sorted(groups, key=lambda item: item.reference.window_min_nm)


def _component_from_result(result: LineFitResult) -> FitComponentSpec:
    return {
        "label": result.line_id,
        "amplitude": result.amplitude,
        "center_nm": result.center_nm,
        "sigma_nm": result.sigma_nm,
        "rest_wavelength_nm": result.rest_wavelength_nm,
    }


def _safe_line_group_name(result: LineFitResult) -> str:
    name = result.blend_group_id or result.line_id
    return name.replace("+", "__").replace(",", "__").replace("/", "-")


def _plot_line_fit_panel(
    spectrum: Spectrum,
    result: LineFitResult,
    component_specs: list[FitComponentSpec],
    *,
    ax,
) -> None:
    window = spectrum.window(result.window_min_nm, result.window_max_nm)
    x = window.wavelength_nm
    y = window.intensity
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    x_model = np.linspace(result.window_min_nm, result.window_max_nm, 360)

    ax.plot(x, y, ":+", lw=0.75, ms=3.0, mew=0.65, color="black", label="data")
    if np.isfinite(result.baseline_offset):
        ax.axhline(result.baseline_offset, color=BASELINE_COLOR, lw=0.7, ls=":")

    profiles = []
    for spec in component_specs:
        if not _finite_component(spec):
            continue
        label = str(spec.get("label", "component"))
        color = _component_color(spec, label)
        profile = gaussian_area_model(
            x_model,
            float(spec["amplitude"]),
            float(spec["center_nm"]),
            float(spec["sigma_nm"]),
        )
        profiles.append(profile)
        is_target = label == result.line_id
        ax.plot(
            x_model,
            result.baseline_offset + profile,
            lw=0.95 if is_target else 0.8,
            ls="-" if is_target else "--",
            alpha=0.92,
            color=color,
        )
        ax.fill_between(
            x_model,
            result.baseline_offset,
            result.baseline_offset + profile,
            facecolor="none",
            edgecolor=color,
            hatch="////",
            linewidth=0.0,
            alpha=0.20,
        )
        _label_component(ax, x_model, result.baseline_offset + profile, spec, label, color)
        rest_wavelength = _component_rest_wavelength(spec, result)
        if np.isfinite(rest_wavelength):
            _plot_short_tick(
                ax,
                rest_wavelength,
                color=color,
                y0=0.04,
                y1=0.18,
                lw=0.75,
                ls=":",
            )

    if profiles:
        y_sum_raw = result.baseline_offset + np.sum(profiles, axis=0)
        ax.plot(x_model, y_sum_raw, lw=1.15, color=FIT_SUM_COLOR)

    ax.set_title(
        _fit_panel_title(result, component_specs),
        loc="left",
        fontsize=8,
        pad=2,
        y=1.08,
    )
    _plot_fit_panel_status_text(ax, result)
    set_inward_ticks(ax)
    ax.tick_params(labelsize=7)
    ax.margins(x=0.02, y=0.12)


def _finite_component(spec: FitComponentSpec) -> bool:
    keys = ("amplitude", "center_nm", "sigma_nm")
    return all(key in spec and np.isfinite(float(spec[key])) for key in keys)


def _fit_panel_title(
    result: LineFitResult,
    component_specs: list[FitComponentSpec],
) -> str:
    title = _fit_title(result, component_specs)
    if "H2 " in title:
        title = title.replace("H2 ", "", 1)
    return title


def _fit_panel_status(result: LineFitResult) -> str:
    parts = []
    if result.legacy_matrix_action == "zero":
        parts.append("legacy zero")
    elif result.legacy_policy in {"suspicious", "accept_with_warning"}:
        parts.append(result.legacy_policy.replace("_", " "))
    if np.isfinite(result.center_offset_from_rest_nm):
        parts.append(f"dx={result.center_offset_from_rest_nm:+.3g} nm")
    if np.isfinite(result.relative_error):
        parts.append(f"err={result.relative_error:.2g}")
    if "unresolved_coincident_database_lines" in result.status:
        parts.append("unresolved")
    elif "bound" in result.status:
        parts.append("bound")
    elif not result.success:
        parts.append("failed")
    return " | ".join(parts)


def _plot_fit_panel_status_text(ax, result: LineFitResult) -> None:
    status = _fit_panel_status(result)
    if not status:
        return
    ax.text(
        0.985,
        1.01,
        status,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="0.35",
        clip_on=False,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.4},
    )


def _plot_neighbor_lines(
    ax,
    result: LineFitResult,
    neighbor_lines: list[FulcherLine],
) -> None:
    for line in neighbor_lines:
        if line.line_id == result.line_id:
            continue
        if line.line_id in result.close_neighbor_ids.split(","):
            continue
        if not (result.window_min_nm <= line.wavelength_nm <= result.window_max_nm):
            continue
        color = LINE_COLORS.get(line.band, "0.5")
        _plot_short_tick(ax, line.wavelength_nm, color=color, y0=0.77, y1=0.9, lw=0.8)
        _plot_line_label(ax, line.wavelength_nm, f"Q{line.N}({line.band})", color=color)


def _plot_line_label(
    ax,
    wavelength_nm: float,
    label: str,
    *,
    color: str,
    y_fraction: float = 0.94,
) -> None:
    ymin, ymax = ax.get_ylim()
    y_text = ymin + y_fraction * (ymax - ymin)
    ax.text(
        wavelength_nm,
        y_text,
        label,
        rotation=0,
        va="bottom",
        ha="center",
        fontsize=7,
        color=color,
    )


def _selector_label(spectrum: Spectrum) -> str:
    if not spectrum.selectors:
        return ""
    return " ".join(f"{key}={value}" for key, value in spectrum.selectors.items())


def _line_color_from_label(label: str) -> str:
    for band, color in LINE_COLORS.items():
        if band in label:
            return color
    return "0.5"


def _spectrum_label(spectrum: Spectrum) -> str:
    label = spectrum.shot_id or spectrum.source_path.stem
    selectors = _selector_label(spectrum)
    if selectors:
        label = f"{label} {selectors}"
    return label


def _plot_band_rails(
    ax,
    lines: list[FulcherLine],
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    *,
    label_lines: list[FulcherLine],
    show_all_line_labels: bool,
    labels_above_axes: bool,
) -> None:
    visible = [
        line
        for line in lines
        if wavelength_min_nm <= line.wavelength_nm <= wavelength_max_nm
    ]
    if not visible:
        return
    labelled_ids = {line.line_id for line in label_lines}
    transform = ax.get_xaxis_transform()
    by_band = sorted({line.band for line in visible})
    rail_base = 1.12 if labels_above_axes else 0.93
    rail_gap = 0.13 if labels_above_axes else 0.09
    for band_index, band in enumerate(by_band):
        band_lines = [line for line in visible if line.band == band]
        row_y = rail_base + band_index * rail_gap if labels_above_axes else rail_base - band_index * rail_gap
        rail_y = row_y - 0.035
        color = band_color(band)
        x0 = min(line.wavelength_nm for line in band_lines)
        x1 = max(line.wavelength_nm for line in band_lines)
        ax.hlines(
            rail_y,
            x0,
            x1,
            color=color,
            lw=1.1,
            transform=transform,
            clip_on=False,
        )
        ax.text(
            x0 - 0.25,
            rail_y,
            f"(v'-v'')=({band})",
            ha="right",
            va="center",
            fontsize=9,
            color="black",
            transform=transform,
            clip_on=False,
        )
        for line in band_lines:
            ax.vlines(
                line.wavelength_nm,
                rail_y - 0.04,
                rail_y,
                color=color,
                lw=0.8,
                transform=transform,
                clip_on=False,
            )
        _label_selected_rail_lines(
            ax,
            band_lines,
            labelled_ids,
            row_y,
            transform,
            show_all=show_all_line_labels,
        )


def _label_selected_rail_lines(
    ax,
    lines,
    labelled_ids: set[str],
    y: float,
    transform,
    *,
    show_all: bool,
) -> None:
    for line in sorted(lines, key=lambda item: item.wavelength_nm):
        if not show_all and line.line_id not in labelled_ids:
            continue
        ax.text(
            line.wavelength_nm,
            y,
            f"Q{line.N}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            transform=transform,
            clip_on=False,
        )


def _plot_region_guides(
    ax,
    lines: list[FulcherLine],
    wavelength_min_nm: float,
    wavelength_max_nm: float,
) -> None:
    seen: set[float] = set()
    for line in lines:
        wavelength = line.wavelength_nm
        if not (wavelength_min_nm <= wavelength <= wavelength_max_nm):
            continue
        rounded = round(wavelength, 5)
        if rounded in seen:
            continue
        seen.add(rounded)
        ax.axvline(
            wavelength,
            color=band_color(line.band),
            lw=1.0,
            alpha=0.5,
            zorder=0,
        )


def _plot_component_markers(
    ax,
    result: LineFitResult,
    component_specs: list[dict[str, float | str]],
    *,
    show_database: bool,
    show_fitted: bool,
) -> None:
    for spec in component_specs:
        label = str(spec.get("label", "component"))
        color = _component_color(spec, label)
        rest_wavelength = _component_rest_wavelength(spec, result)
        fit_center = float(spec["center_nm"]) if "center_nm" in spec else float("nan")
        if show_database and np.isfinite(rest_wavelength):
            _plot_short_tick(
                ax,
                rest_wavelength,
                color=color,
                y0=0.05,
                y1=0.17,
                lw=0.95,
                ls=":",
                alpha=0.95,
            )
            ax.text(
                rest_wavelength,
                _axis_fraction_to_data_y(ax, 0.18),
                "db",
                ha="center",
                va="bottom",
                fontsize=7,
                color=color,
            )
        if show_fitted and np.isfinite(fit_center):
            _plot_short_tick(
                ax,
                fit_center,
                color=color,
                y0=0.05,
                y1=0.26,
                lw=1.15,
                ls="-",
                alpha=0.95,
            )
            ax.text(
                fit_center,
                _axis_fraction_to_data_y(ax, 0.27),
                "fit",
                ha="center",
                va="bottom",
                fontsize=7,
                color=color,
            )


def _plot_short_tick(
    ax,
    wavelength_nm: float,
    *,
    color: str,
    y0: float,
    y1: float,
    lw: float,
    ls: str = "-",
    alpha: float = 0.9,
) -> None:
    ax.vlines(
        wavelength_nm,
        _axis_fraction_to_data_y(ax, y0),
        _axis_fraction_to_data_y(ax, y1),
        color=color,
        lw=lw,
        ls=ls,
        alpha=alpha,
    )


def _axis_fraction_to_data_y(ax, fraction: float) -> float:
    ymin, ymax = ax.get_ylim()
    return ymin + fraction * (ymax - ymin)


def _label_component(
    ax,
    x_model: np.ndarray,
    y_model: np.ndarray,
    spec: dict[str, float | str],
    label: str,
    color: str,
) -> None:
    if not np.any(np.isfinite(y_model)):
        return
    peak_index = int(np.nanargmax(y_model))
    ax.text(
        x_model[peak_index],
        y_model[peak_index],
        _component_display_label(spec, label),
        ha="center",
        va="bottom",
        fontsize=8,
        color=color,
    )


def _plot_fit_note(ax, result: LineFitResult) -> None:
    lines = [
        f"target: {compact_line_label(result.line_id)}",
        f"center-rest: {result.center_offset_from_rest_nm:.4g} nm",
        f"sigma: {result.sigma_nm:.4g} nm",
        f"area: {result.amplitude:.4g}",
        f"rms: {result.residual_rms:.3g}",
    ]
    if result.blend_component_count > 1:
        lines.insert(1, f"blend: {result.blend_component_count} components")
    if "unresolved_coincident_database_lines" in result.status:
        lines.insert(1, "unresolved: no spectral split")
    if result.legacy_policy:
        lines.append(f"legacy: {result.legacy_policy.replace('_', ' ')}")
    if result.legacy_matrix_action == "zero":
        lines.append("matrix export: zero")
    ax.text(
        0.015,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.86, "pad": 3},
    )


def _should_hatch_component() -> bool:
    return True


def _component_color(spec: dict[str, float | str], label: str) -> str:
    if "color" in spec:
        return str(spec["color"])
    if _is_contaminant_component(spec):
        return "0.5"
    return _line_color_from_label(label)


def _is_contaminant_component(spec: dict[str, float | str]) -> bool:
    role = str(spec.get("role", "")).lower()
    return role in {"contaminant", "contamination", "blended"} or bool(
        spec.get("contaminant", False)
    )


def _component_display_label(spec: dict[str, float | str], label: str) -> str:
    if "display_label" in spec:
        return str(spec["display_label"])
    if _is_contaminant_component(spec):
        return "blended"
    return compact_line_label(label).replace("H2 ", "")


def _component_rest_wavelength(
    spec: dict[str, float | str],
    result: LineFitResult,
) -> float:
    if "rest_wavelength_nm" in spec:
        return float(spec["rest_wavelength_nm"])
    if str(spec.get("label", "")) == result.line_id:
        return result.rest_wavelength_nm
    return float("nan")


def _fit_title(
    result: LineFitResult,
    component_specs: list[dict[str, float | str]],
) -> str:
    assigned_labels = [
        _component_display_label(spec, str(spec.get("label", "component")))
        for spec in component_specs
        if not _is_contaminant_component(spec)
    ]
    if not assigned_labels:
        return compact_line_label(result.line_id)
    unique_labels = list(dict.fromkeys(assigned_labels))
    if len(unique_labels) == 1:
        return f"H2 {unique_labels[0]}" if not unique_labels[0].startswith("H2 ") else unique_labels[0]
    return "H2 " + " + ".join(unique_labels)
