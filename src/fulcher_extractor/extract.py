"""High-level Fulcher-alpha extraction workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .fit import (
    ContaminantSpec,
    FitConfig,
    LineFitResult,
    fit_line_group,
    fit_line_with_contaminants,
    group_close_lines,
)
from .line_database import FulcherLine, filter_lines, load_lines
from .line_policy import apply_line_policies, load_line_policies
from .spectrocube_io import Spectrum


@dataclass(frozen=True)
class DecontaminationRule:
    """Explicit non-database contaminant model for one target line."""

    line_id: str
    contaminants: tuple[ContaminantSpec, ...]
    line_left_width_nm: float | None = None
    line_right_width_nm: float | None = None
    status_tags: tuple[str, ...] = ()


DEFAULT_DECONTAMINATION_RULES = {
    "H2_Q7_0-0": DecontaminationRule(
        line_id="H2_Q7_0-0",
        contaminants=(
            ContaminantSpec(
                label="blue_shoulder",
                initial_offset_nm=-0.055,
                lower_offset_nm=-0.12,
                upper_offset_nm=-0.015,
            ),
            ContaminantSpec(
                label="red_shoulder",
                initial_offset_nm=0.055,
                lower_offset_nm=0.015,
                upper_offset_nm=0.12,
            ),
        ),
    ),
    "H2_Q4_1-1": DecontaminationRule(
        line_id="H2_Q4_1-1",
        contaminants=(
            ContaminantSpec(
                label="red_shoulder_neighbor",
                initial_offset_nm=0.075,
                lower_offset_nm=0.075,
                upper_offset_nm=0.075,
                fixed_offset_nm=0.075,
            ),
        ),
        line_left_width_nm=0.07,
        line_right_width_nm=0.08,
        status_tags=("suspicious_decontamination",),
    ),
    "H2_Q6_1-1": DecontaminationRule(
        line_id="H2_Q6_1-1",
        contaminants=(
            ContaminantSpec(
                label="blue_neighbor_1",
                initial_offset_nm=-0.145,
                lower_offset_nm=-0.18,
                upper_offset_nm=-0.105,
            ),
            ContaminantSpec(
                label="blue_neighbor_2",
                initial_offset_nm=-0.075,
                lower_offset_nm=-0.105,
                upper_offset_nm=-0.040,
            ),
            ContaminantSpec(
                label="red_neighbor",
                initial_offset_nm=0.100,
                lower_offset_nm=0.060,
                upper_offset_nm=0.130,
            ),
        ),
    ),
}


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
    decontamination_rules = DEFAULT_DECONTAMINATION_RULES
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
        if len(group) == 1 and group[0].line_id in decontamination_rules:
            rule = decontamination_rules[group[0].line_id]
            result = fit_line_with_contaminants(
                spectrum,
                group[0],
                list(rule.contaminants),
                config=_config_for_rule(cfg, rule),
            )
            results.append(_apply_rule_status_tags(result, rule))
        else:
            results.extend(fit_line_group(spectrum, group, config=cfg))
    return apply_line_policies(results, policies=load_line_policies())


def _config_for_rule(cfg: FitConfig, rule: DecontaminationRule) -> FitConfig:
    overrides = {}
    if rule.line_left_width_nm is not None:
        overrides["line_left_width_nm"] = rule.line_left_width_nm
    if rule.line_right_width_nm is not None:
        overrides["line_right_width_nm"] = rule.line_right_width_nm
    return replace(cfg, **overrides) if overrides else cfg


def _apply_rule_status_tags(
    result: LineFitResult,
    rule: DecontaminationRule,
) -> LineFitResult:
    status = result.status
    for tag in rule.status_tags:
        if tag not in status.split(";"):
            status = f"{status};{tag}"
    return replace(result, status=status) if status != result.status else result
