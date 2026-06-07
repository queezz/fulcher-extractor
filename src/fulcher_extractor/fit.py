"""Line fitting routines and uncertainty estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import curve_fit

from .line_database import FulcherLine
from .line_models import gaussian_area_model, local_minimum_baseline
from .spectrocube_io import Spectrum


@dataclass(frozen=True)
class FitConfig:
    line_half_width_nm: float = 0.18
    line_left_width_nm: float | None = None
    line_right_width_nm: float | None = None
    background_half_width_nm: float = 1.5
    global_dx_nm: float = 0.0
    center_offset_nm: float = 0.08
    center_left_offset_nm: float | None = None
    center_right_offset_nm: float | None = None
    sigma_min_nm: float = 0.015
    sigma_max_nm: float = 0.16
    instrument_sigma_nm: float | None = None
    instrument_sigma_leeway_nm: float = 0.015
    initial_sigma_nm: float = 0.055
    baseline_strategy: str = "local_minimum"
    close_neighbor_threshold_nm: float | None = 0.15


@dataclass(frozen=True)
class LineFitResult:
    line_id: str
    isotopologue: str
    branch: str
    band: str
    N: int
    rest_wavelength_nm: float
    amplitude: float
    amplitude_stderr: float
    center_nm: float
    center_stderr_nm: float
    sigma_nm: float
    sigma_stderr_nm: float
    fwhm_nm: float
    baseline_offset: float
    baseline_strategy: str
    window_min_nm: float
    window_max_nm: float
    background_min_nm: float
    background_max_nm: float
    n_points: int
    residual_rms: float
    success: bool
    status: str
    background_min_intensity: float = float("nan")
    background_median_intensity: float = float("nan")
    background_max_intensity: float = float("nan")
    global_dx_nm: float = 0.0
    expected_center_nm: float = float("nan")
    center_offset_from_rest_nm: float = float("nan")
    center_offset_from_expected_nm: float = float("nan")
    sigma_lower_bound_nm: float = float("nan")
    sigma_upper_bound_nm: float = float("nan")
    center_lower_bound_nm: float = float("nan")
    center_upper_bound_nm: float = float("nan")
    blend_group_id: str = ""
    blend_component_count: int = 1
    close_neighbor_ids: str = ""
    blend_delta_nm: float = float("nan")
    legacy_policy: str = ""
    legacy_matrix_action: str = ""
    legacy_evidence: str = ""
    legacy_fit_hint: str = ""
    legacy_line_scale_role: str = ""
    matrix_amplitude: float = float("nan")
    matrix_amplitude_stderr: float = float("nan")
    contaminant_component_count: int = 0
    contaminant_labels: str = ""
    contaminant_amplitudes: str = ""
    contaminant_centers_nm: str = ""
    contaminant_sigmas_nm: str = ""

    @property
    def relative_error(self) -> float:
        if self.amplitude == 0 or not np.isfinite(self.amplitude_stderr):
            return float("nan")
        return abs(self.amplitude_stderr / self.amplitude)


@dataclass(frozen=True)
class InstrumentWidthEstimate:
    sigma_nm: float
    fwhm_nm: float
    n_lines: int
    sigma_q10_nm: float
    sigma_q90_nm: float


@dataclass(frozen=True)
class ContaminantSpec:
    """One non-database contaminant component in a decontaminating line fit."""

    label: str
    initial_offset_nm: float
    lower_offset_nm: float
    upper_offset_nm: float


def _finite_window(wavelength_nm: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(wavelength_nm) & np.isfinite(intensity)
    return wavelength_nm[mask], intensity[mask]


def fit_single_line(
    spectrum: Spectrum,
    line: FulcherLine,
    *,
    config: FitConfig | None = None,
) -> LineFitResult:
    """Fit one Fulcher line with explicit local-minimum baseline reporting."""
    return _fit_single_line_core(spectrum, line, config or FitConfig())


def fit_line_group(
    spectrum: Spectrum,
    lines: list[FulcherLine],
    *,
    config: FitConfig | None = None,
) -> list[LineFitResult]:
    """Fit one line or a close-neighbour group as a shared-width Gaussian sum."""
    if len(lines) == 1:
        return [fit_single_line(spectrum, lines[0], config=config)]

    cfg = config or FitConfig()
    ordered = sorted(lines, key=lambda line: line.wavelength_nm)
    expected = np.array([line.wavelength_nm + cfg.global_dx_nm for line in ordered])
    line_left, line_right = _line_window_widths(cfg)
    center_left, center_right = _center_offsets(cfg)
    line_min = float(expected.min() - line_left)
    line_max = float(expected.max() + line_right)
    background_min = float(expected.min() - cfg.background_half_width_nm)
    background_max = float(expected.max() + cfg.background_half_width_nm)
    background_window = spectrum.window(background_min, background_max)
    _, background_y = _finite_window(
        background_window.wavelength_nm, background_window.intensity
    )
    background_stats = _background_stats(background_y)
    window = spectrum.window(line_min, line_max)
    x, y_raw = _finite_window(window.wavelength_nm, window.intensity)
    group_id = "+".join(line.line_id for line in ordered)
    close_ids = ",".join(line.line_id for line in ordered)

    if x.size < 5:
        return [
            _failed_result(
                line,
                cfg,
                line_min,
                line_max,
                background_min,
                background_max,
                "too_few_points;blend_group",
                background_stats,
                blend_group_id=group_id,
                blend_component_count=len(ordered),
                close_neighbor_ids=close_ids,
            )
            for line in ordered
        ]

    if cfg.baseline_strategy != "local_minimum":
        raise ValueError(f"Unsupported baseline strategy: {cfg.baseline_strategy!r}")
    baseline = local_minimum_baseline(y_raw)
    y = y_raw - baseline
    sigma_min, sigma_max, sigma_initial = _sigma_bounds(cfg)

    peak_area = max(
        float(np.trapezoid(np.clip(y, 0.0, None), x)),
        float(np.nanmax(y)) * sigma_initial * np.sqrt(2.0 * np.pi),
        np.finfo(float).eps,
    )
    weights = _relative_band_weights(ordered)
    initial_areas = peak_area * weights / weights.sum()
    p0 = list(initial_areas) + [0.0, sigma_initial]
    lower = [0.0] * len(ordered)
    lower += [-center_left, sigma_min]
    upper = [np.inf] * len(ordered)
    upper += [center_right, sigma_max]

    coincident = _has_coincident_database_lines(ordered)
    try:
        popt, pcov = curve_fit(
            lambda wavelengths, *params: _blend_model(
                wavelengths, len(ordered), expected, params
            ),
            x,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=40000,
        )
        y_fit = _blend_model(x, len(ordered), expected, popt)
        residual = y - y_fit
        perr = np.sqrt(np.diag(pcov)) if pcov.size else np.full(len(popt), np.nan)
        status = "ok;blend_group"
        success = True
        sigma = float(popt[-1])
        if abs(sigma - sigma_max) <= 1e-8:
            status = f"{status};sigma_at_upper_bound"
        if abs(sigma - sigma_min) <= 1e-8:
            status = f"{status};sigma_at_lower_bound"
        if coincident:
            status = f"{status};unresolved_coincident_database_lines"
    except Exception as exc:
        popt = np.full(len(ordered) + 2, np.nan)
        perr = np.full(len(ordered) + 2, np.nan)
        residual = np.full_like(y, np.nan)
        status = f"fit_failed:{type(exc).__name__};blend_group"
        success = False

    sigma = float(popt[-1]) if np.isfinite(popt[-1]) else float("nan")
    blend_delta = float(popt[-2]) if np.isfinite(popt[-2]) else float("nan")
    results = []
    for idx, line in enumerate(ordered):
        center = float(expected[idx] + blend_delta)
        center_status = status
        if np.isfinite(blend_delta) and abs(blend_delta - lower[len(ordered)]) <= 1e-8:
            center_status = f"{center_status};center_at_lower_bound"
        if np.isfinite(blend_delta) and abs(blend_delta - upper[len(ordered)]) <= 1e-8:
            center_status = f"{center_status};center_at_upper_bound"
        results.append(
            LineFitResult(
                line_id=line.line_id,
                isotopologue=line.isotopologue,
                branch=line.branch,
                band=line.band,
                N=line.N,
                rest_wavelength_nm=line.wavelength_nm,
                amplitude=float(popt[idx]),
                amplitude_stderr=float(perr[idx]),
                center_nm=center,
                center_stderr_nm=float(perr[-2]),
                sigma_nm=sigma,
                sigma_stderr_nm=float(perr[-1]),
                fwhm_nm=float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma),
                baseline_offset=float(baseline),
                baseline_strategy=cfg.baseline_strategy,
                window_min_nm=line_min,
                window_max_nm=line_max,
                background_min_nm=background_min,
                background_max_nm=background_max,
                n_points=int(x.size),
                residual_rms=float(np.sqrt(np.nanmean(residual**2))),
                success=success,
                status=center_status,
                background_min_intensity=background_stats[0],
                background_median_intensity=background_stats[1],
                background_max_intensity=background_stats[2],
                global_dx_nm=cfg.global_dx_nm,
                expected_center_nm=float(expected[idx]),
                center_offset_from_rest_nm=float(center - line.wavelength_nm),
                center_offset_from_expected_nm=float(center - expected[idx]),
                sigma_lower_bound_nm=sigma_min,
                sigma_upper_bound_nm=sigma_max,
                center_lower_bound_nm=float(expected[idx] - center_left),
                center_upper_bound_nm=float(expected[idx] + center_right),
                blend_group_id=group_id,
                blend_component_count=len(ordered),
                close_neighbor_ids=close_ids,
                blend_delta_nm=blend_delta,
            )
        )
    return results


def fit_line_with_contaminants(
    spectrum: Spectrum,
    line: FulcherLine,
    contaminants: list[ContaminantSpec],
    *,
    config: FitConfig | None = None,
) -> LineFitResult:
    """Fit one target line together with explicit non-database contaminants."""
    if not contaminants:
        return fit_single_line(spectrum, line, config=config)

    cfg = config or FitConfig()
    expected_center = line.wavelength_nm + cfg.global_dx_nm
    line_left, line_right = _line_window_widths(cfg)
    center_left, center_right = _center_offsets(cfg)
    line_min = min(
        [expected_center - line_left]
        + [expected_center + spec.lower_offset_nm - line_left for spec in contaminants]
    )
    line_max = max(
        [expected_center + line_right]
        + [expected_center + spec.upper_offset_nm + line_right for spec in contaminants]
    )
    background_min = expected_center - cfg.background_half_width_nm
    background_max = expected_center + cfg.background_half_width_nm
    background_window = spectrum.window(background_min, background_max)
    _, background_y = _finite_window(
        background_window.wavelength_nm, background_window.intensity
    )
    background_stats = _background_stats(background_y)
    window = spectrum.window(line_min, line_max)
    x, y_raw = _finite_window(window.wavelength_nm, window.intensity)

    component_count = 1 + len(contaminants)
    contaminant_labels = ",".join(spec.label for spec in contaminants)
    group_id = f"{line.line_id}+{contaminant_labels}"
    if x.size < 5:
        return _failed_result(
            line,
            cfg,
            line_min,
            line_max,
            background_min,
            background_max,
            "too_few_points;decontaminated",
            background_stats,
            blend_group_id=group_id,
            blend_component_count=component_count,
            close_neighbor_ids=contaminant_labels,
        )

    if cfg.baseline_strategy != "local_minimum":
        raise ValueError(f"Unsupported baseline strategy: {cfg.baseline_strategy!r}")
    baseline = local_minimum_baseline(y_raw)
    y = y_raw - baseline
    sigma_min, sigma_max, sigma_initial = _sigma_bounds(cfg)
    target_lower = expected_center - center_left
    target_upper = expected_center + center_right

    total_area = max(
        float(np.trapezoid(np.clip(y, 0.0, None), x)),
        float(np.nanmax(y)) * sigma_initial * np.sqrt(2.0 * np.pi),
        np.finfo(float).eps,
    )
    contaminant_initial_area = total_area * 0.12
    target_initial_area = max(
        total_area - contaminant_initial_area * len(contaminants),
        total_area * 0.2,
    )
    p0 = [target_initial_area]
    p0.extend([contaminant_initial_area] * len(contaminants))
    p0.append(expected_center)
    p0.extend(expected_center + spec.initial_offset_nm for spec in contaminants)
    p0.append(sigma_initial)

    lower = [0.0] * component_count
    lower.append(target_lower)
    lower.extend(expected_center + spec.lower_offset_nm for spec in contaminants)
    lower.append(sigma_min)
    upper = [np.inf] * component_count
    upper.append(target_upper)
    upper.extend(expected_center + spec.upper_offset_nm for spec in contaminants)
    upper.append(sigma_max)

    try:
        popt, pcov = curve_fit(
            lambda wavelengths, *params: _independent_component_model(
                wavelengths, component_count, params
            ),
            x,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=60000,
        )
        y_fit = _independent_component_model(x, component_count, popt)
        residual = y - y_fit
        perr = np.sqrt(np.diag(pcov)) if pcov.size else np.full(len(popt), np.nan)
        status = "ok;decontaminated;contaminant_components"
        success = True
        sigma = float(popt[-1])
        if abs(sigma - sigma_max) <= 1e-8:
            status = f"{status};sigma_at_upper_bound"
        if abs(sigma - sigma_min) <= 1e-8:
            status = f"{status};sigma_at_lower_bound"
        target_center = float(popt[component_count])
        if abs(target_center - target_lower) <= 1e-8:
            status = f"{status};center_at_lower_bound"
        if abs(target_center - target_upper) <= 1e-8:
            status = f"{status};center_at_upper_bound"
    except Exception as exc:
        popt = np.full(2 * component_count + 1, np.nan)
        perr = np.full(2 * component_count + 1, np.nan)
        residual = np.full_like(y, np.nan)
        status = f"fit_failed:{type(exc).__name__};decontaminated"
        success = False

    sigma = float(popt[-1]) if np.isfinite(popt[-1]) else float("nan")
    target_center = float(popt[component_count])
    contaminant_areas = [float(value) for value in popt[1:component_count]]
    contaminant_centers = [
        float(value) for value in popt[component_count + 1 : 2 * component_count]
    ]
    return LineFitResult(
        line_id=line.line_id,
        isotopologue=line.isotopologue,
        branch=line.branch,
        band=line.band,
        N=line.N,
        rest_wavelength_nm=line.wavelength_nm,
        amplitude=float(popt[0]),
        amplitude_stderr=float(perr[0]),
        center_nm=target_center,
        center_stderr_nm=float(perr[component_count]),
        sigma_nm=sigma,
        sigma_stderr_nm=float(perr[-1]),
        fwhm_nm=float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma),
        baseline_offset=float(baseline),
        baseline_strategy=cfg.baseline_strategy,
        window_min_nm=line_min,
        window_max_nm=line_max,
        background_min_nm=background_min,
        background_max_nm=background_max,
        n_points=int(x.size),
        residual_rms=float(np.sqrt(np.nanmean(residual**2))),
        success=success,
        status=status,
        background_min_intensity=background_stats[0],
        background_median_intensity=background_stats[1],
        background_max_intensity=background_stats[2],
        global_dx_nm=cfg.global_dx_nm,
        expected_center_nm=expected_center,
        center_offset_from_rest_nm=float(target_center - line.wavelength_nm),
        center_offset_from_expected_nm=float(target_center - expected_center),
        sigma_lower_bound_nm=sigma_min,
        sigma_upper_bound_nm=sigma_max,
        center_lower_bound_nm=target_lower,
        center_upper_bound_nm=target_upper,
        blend_group_id=group_id,
        blend_component_count=component_count,
        close_neighbor_ids=contaminant_labels,
        blend_delta_nm=float("nan"),
        contaminant_component_count=len(contaminants),
        contaminant_labels=contaminant_labels,
        contaminant_amplitudes=_join_floats(contaminant_areas),
        contaminant_centers_nm=_join_floats(contaminant_centers),
        contaminant_sigmas_nm=_join_floats([sigma] * len(contaminants)),
    )


def group_close_lines(
    lines: list[FulcherLine], threshold_nm: float
) -> list[list[FulcherLine]]:
    """Connected components of database lines closer than *threshold_nm*."""
    ordered = sorted(lines, key=lambda line: line.wavelength_nm)
    groups: list[list[FulcherLine]] = []
    current: list[FulcherLine] = []
    for line in ordered:
        if not current:
            current = [line]
            continue
        if line.wavelength_nm - current[-1].wavelength_nm <= threshold_nm:
            current.append(line)
        else:
            groups.append(current)
            current = [line]
    if current:
        groups.append(current)
    return groups


def _fit_single_line_core(
    spectrum: Spectrum,
    line: FulcherLine,
    cfg: FitConfig,
) -> LineFitResult:
    expected_center = line.wavelength_nm + cfg.global_dx_nm
    line_left, line_right = _line_window_widths(cfg)
    center_left, center_right = _center_offsets(cfg)
    line_min = expected_center - line_left
    line_max = expected_center + line_right
    background_min = expected_center - cfg.background_half_width_nm
    background_max = expected_center + cfg.background_half_width_nm
    background_window = spectrum.window(background_min, background_max)
    _, background_y = _finite_window(
        background_window.wavelength_nm, background_window.intensity
    )
    background_stats = _background_stats(background_y)
    window = spectrum.window(line_min, line_max)
    x, y_raw = _finite_window(window.wavelength_nm, window.intensity)

    if x.size < 5:
        return _failed_result(
            line,
            cfg,
            line_min,
            line_max,
            background_min,
            background_max,
            "too_few_points",
            background_stats,
        )

    if cfg.baseline_strategy != "local_minimum":
        raise ValueError(f"Unsupported baseline strategy: {cfg.baseline_strategy!r}")
    baseline = local_minimum_baseline(y_raw)
    y = y_raw - baseline
    area0 = max(float(np.trapezoid(np.clip(y, 0.0, None), x)), np.finfo(float).eps)
    peak = max(float(np.nanmax(y)), np.finfo(float).eps)
    area0 = max(area0, peak * cfg.initial_sigma_nm * np.sqrt(2.0 * np.pi) * 0.5)

    sigma_min, sigma_max, sigma_initial = _sigma_bounds(cfg)
    center_lower = expected_center - center_left
    center_upper = expected_center + center_right
    lower = [0.0, center_lower, sigma_min]
    upper = [np.inf, center_upper, sigma_max]
    p0 = [area0, expected_center, sigma_initial]

    try:
        popt, pcov = curve_fit(
            gaussian_area_model,
            x,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
        y_fit = gaussian_area_model(x, *popt)
        residual = y - y_fit
        perr = np.sqrt(np.diag(pcov)) if pcov.size else np.full(3, np.nan)
        status = "ok"
        success = True
        if abs(float(popt[2]) - sigma_max) <= 1e-8:
            status = "ok;sigma_at_upper_bound"
        if abs(float(popt[2]) - sigma_min) <= 1e-8:
            status = "ok;sigma_at_lower_bound"
        if abs(float(popt[1]) - lower[1]) <= 1e-8:
            status = f"{status};center_at_lower_bound"
        if abs(float(popt[1]) - upper[1]) <= 1e-8:
            status = f"{status};center_at_upper_bound"
    except Exception as exc:
        popt = np.array([np.nan, np.nan, np.nan])
        perr = np.array([np.nan, np.nan, np.nan])
        residual = np.full_like(y, np.nan)
        status = f"fit_failed:{type(exc).__name__}"
        success = False

    sigma = float(popt[2]) if np.isfinite(popt[2]) else float("nan")
    return LineFitResult(
        line_id=line.line_id,
        isotopologue=line.isotopologue,
        branch=line.branch,
        band=line.band,
        N=line.N,
        rest_wavelength_nm=line.wavelength_nm,
        amplitude=float(popt[0]),
        amplitude_stderr=float(perr[0]),
        center_nm=float(popt[1]),
        center_stderr_nm=float(perr[1]),
        sigma_nm=sigma,
        sigma_stderr_nm=float(perr[2]),
        fwhm_nm=float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma),
        baseline_offset=float(baseline),
        baseline_strategy=cfg.baseline_strategy,
        window_min_nm=line_min,
        window_max_nm=line_max,
        background_min_nm=background_min,
        background_max_nm=background_max,
        n_points=int(x.size),
        residual_rms=float(np.sqrt(np.nanmean(residual**2))),
        success=success,
        status=status,
        background_min_intensity=background_stats[0],
        background_median_intensity=background_stats[1],
        background_max_intensity=background_stats[2],
        global_dx_nm=cfg.global_dx_nm,
        expected_center_nm=expected_center,
        center_offset_from_rest_nm=float(popt[1] - line.wavelength_nm),
        center_offset_from_expected_nm=float(popt[1] - expected_center),
        sigma_lower_bound_nm=sigma_min,
        sigma_upper_bound_nm=sigma_max,
        center_lower_bound_nm=center_lower,
        center_upper_bound_nm=center_upper,
    )


def estimate_instrument_width(
    results: Iterable[LineFitResult],
    *,
    max_relative_error: float = 0.08,
    max_sigma_nm: float = 0.08,
    exclude_bound_limited: bool = True,
) -> InstrumentWidthEstimate:
    """Estimate instrumental Gaussian width from high-confidence fitted lines."""
    sigmas = []
    for result in results:
        if not result.success:
            continue
        if not np.isfinite(result.sigma_nm) or result.sigma_nm <= 0:
            continue
        if result.sigma_nm > max_sigma_nm:
            continue
        if exclude_bound_limited and "bound" in result.status:
            continue
        if result.relative_error > max_relative_error:
            continue
        sigmas.append(result.sigma_nm)
    if not sigmas:
        raise ValueError("No fitted lines passed the instrumental-width filters.")
    sigma = np.asarray(sigmas, dtype=float)
    sigma_med = float(np.median(sigma))
    return InstrumentWidthEstimate(
        sigma_nm=sigma_med,
        fwhm_nm=float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_med),
        n_lines=int(sigma.size),
        sigma_q10_nm=float(np.quantile(sigma, 0.1)),
        sigma_q90_nm=float(np.quantile(sigma, 0.9)),
    )


def _background_stats(y: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(y, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.min(finite)), float(np.median(finite)), float(np.max(finite))


def _blend_model(
    wavelength_nm: np.ndarray,
    component_count: int,
    expected_centers_nm: np.ndarray,
    params,
) -> np.ndarray:
    areas = params[:component_count]
    delta_nm = params[component_count]
    centers = expected_centers_nm + delta_nm
    sigma = params[-1]
    y = np.zeros_like(wavelength_nm, dtype=float)
    for area, center in zip(areas, centers):
        y += gaussian_area_model(wavelength_nm, float(area), float(center), float(sigma))
    return y


def _independent_component_model(
    wavelength_nm: np.ndarray,
    component_count: int,
    params,
) -> np.ndarray:
    areas = params[:component_count]
    centers = params[component_count : 2 * component_count]
    sigma = params[-1]
    y = np.zeros_like(wavelength_nm, dtype=float)
    for area, center in zip(areas, centers):
        y += gaussian_area_model(wavelength_nm, float(area), float(center), float(sigma))
    return y


def _join_floats(values: Iterable[float]) -> str:
    return ",".join("" if not np.isfinite(value) else f"{value:.10g}" for value in values)


def _relative_band_weights(lines: list[FulcherLine]) -> np.ndarray:
    weights_by_band = {"0-0": 1.0, "1-1": 0.45, "2-2": 0.20, "3-3": 0.08}
    return np.array([weights_by_band.get(line.band, 0.1) for line in lines], dtype=float)


def _has_coincident_database_lines(lines: list[FulcherLine]) -> bool:
    wavelengths = sorted(line.wavelength_nm for line in lines)
    return any(
        abs(right - left) < 1e-6 for left, right in zip(wavelengths, wavelengths[1:])
    )


def _sigma_bounds(cfg: FitConfig) -> tuple[float, float, float]:
    if cfg.instrument_sigma_nm is None:
        return cfg.sigma_min_nm, cfg.sigma_max_nm, cfg.initial_sigma_nm
    sigma_min = max(cfg.sigma_min_nm, cfg.instrument_sigma_nm - cfg.instrument_sigma_leeway_nm)
    sigma_max = min(cfg.sigma_max_nm, cfg.instrument_sigma_nm + cfg.instrument_sigma_leeway_nm)
    if sigma_min >= sigma_max:
        raise ValueError(
            "Instrument sigma bounds are invalid: "
            f"sigma_min={sigma_min}, sigma_max={sigma_max}."
        )
    sigma_initial = min(max(cfg.instrument_sigma_nm, sigma_min), sigma_max)
    return sigma_min, sigma_max, sigma_initial


def _line_window_widths(cfg: FitConfig) -> tuple[float, float]:
    left = cfg.line_half_width_nm if cfg.line_left_width_nm is None else cfg.line_left_width_nm
    right = cfg.line_half_width_nm if cfg.line_right_width_nm is None else cfg.line_right_width_nm
    return left, right


def _center_offsets(cfg: FitConfig) -> tuple[float, float]:
    left = cfg.center_offset_nm if cfg.center_left_offset_nm is None else cfg.center_left_offset_nm
    right = cfg.center_offset_nm if cfg.center_right_offset_nm is None else cfg.center_right_offset_nm
    return left, right


def _failed_result(
    line: FulcherLine,
    cfg: FitConfig,
    line_min: float,
    line_max: float,
    background_min: float,
    background_max: float,
    status: str,
    background_stats: tuple[float, float, float] = (float("nan"), float("nan"), float("nan")),
    blend_group_id: str = "",
    blend_component_count: int = 1,
    close_neighbor_ids: str = "",
) -> LineFitResult:
    sigma_min, sigma_max, _ = _sigma_bounds(cfg)
    center_left, center_right = _center_offsets(cfg)
    expected_center = line.wavelength_nm + cfg.global_dx_nm
    return LineFitResult(
        line_id=line.line_id,
        isotopologue=line.isotopologue,
        branch=line.branch,
        band=line.band,
        N=line.N,
        rest_wavelength_nm=line.wavelength_nm,
        amplitude=float("nan"),
        amplitude_stderr=float("nan"),
        center_nm=float("nan"),
        center_stderr_nm=float("nan"),
        sigma_nm=float("nan"),
        sigma_stderr_nm=float("nan"),
        fwhm_nm=float("nan"),
        baseline_offset=float("nan"),
        baseline_strategy=cfg.baseline_strategy,
        window_min_nm=line_min,
        window_max_nm=line_max,
        background_min_nm=background_min,
        background_max_nm=background_max,
        n_points=0,
        residual_rms=float("nan"),
        success=False,
        status=status,
        background_min_intensity=background_stats[0],
        background_median_intensity=background_stats[1],
        background_max_intensity=background_stats[2],
        global_dx_nm=cfg.global_dx_nm,
        expected_center_nm=float("nan"),
        center_offset_from_rest_nm=float("nan"),
        center_offset_from_expected_nm=float("nan"),
        sigma_lower_bound_nm=sigma_min,
        sigma_upper_bound_nm=sigma_max,
        center_lower_bound_nm=expected_center - center_left,
        center_upper_bound_nm=expected_center + center_right,
        blend_group_id=blend_group_id,
        blend_component_count=blend_component_count,
        close_neighbor_ids=close_neighbor_ids,
    )
