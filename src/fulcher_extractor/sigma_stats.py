"""Sigma statistics for clean overview-QC Fulcher lines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .line_database import load_lines
from .line_policy import overview_qc_lines

REJECTED_POLICY_PATTERN = r"reject|suspicious|accept_with_warning|unresolved"
FAILED_STATUS_PATTERN = r"fit_failed|too_few_points|unresolved"


@dataclass(frozen=True)
class SigmaStats:
    """Summary statistics for selected fit-report sigma values."""

    count: int
    median: float
    q10: float
    q90: float
    min: float
    max: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "median": self.median,
            "q10": self.q10,
            "q90": self.q90,
            "min": self.min,
            "max": self.max,
        }


def overview_sigma_line_ids() -> frozenset[str]:
    """Return the display-selected line ids that seed sigma statistics."""
    return frozenset(line.line_id for line in overview_qc_lines(load_lines()))


def clean_sigma_mask(
    fit_table: pd.DataFrame,
    *,
    line_ids: Iterable[str] | None = None,
    bound_tolerance_nm: float = 1e-6,
) -> pd.Series:
    """Return rows appropriate for first-pass clean-line sigma statistics."""
    selected_line_ids = set(line_ids or overview_sigma_line_ids())
    status = _string_column(fit_table, "status")
    legacy_policy = _string_column(fit_table, "legacy_policy")
    sigma = pd.to_numeric(fit_table["sigma_nm"], errors="coerce")
    sigma_lower = pd.to_numeric(fit_table["sigma_lower_bound_nm"], errors="coerce")
    sigma_upper = pd.to_numeric(fit_table["sigma_upper_bound_nm"], errors="coerce")

    success = _bool_column(fit_table, "success")
    finite_sigma = np.isfinite(sigma) & (sigma > 0.0)
    at_lower_bound = (sigma - sigma_lower).abs() <= bound_tolerance_nm
    at_upper_bound = (sigma - sigma_upper).abs() <= bound_tolerance_nm

    return (
        fit_table["line_id"].isin(selected_line_ids)
        & success
        & finite_sigma
        & ~status.str.contains(FAILED_STATUS_PATTERN, regex=True)
        & ~status.str.contains("suspicious_decontamination", regex=False)
        & ~status.str.contains("sigma_at_", regex=False)
        & ~legacy_policy.str.contains(REJECTED_POLICY_PATTERN, regex=True)
        & ~at_lower_bound
        & ~at_upper_bound
    )


def summarize_sigma(fit_table: pd.DataFrame, mask: pd.Series) -> SigmaStats:
    """Summarize sigma_nm values selected by a clean-line mask."""
    sigma = pd.to_numeric(fit_table.loc[mask, "sigma_nm"], errors="coerce").dropna()
    if sigma.empty:
        raise ValueError("No fit-report rows passed the clean sigma mask.")
    quantiles = sigma.quantile([0.1, 0.5, 0.9])
    return SigmaStats(
        count=int(sigma.shape[0]),
        median=float(quantiles.loc[0.5]),
        q10=float(quantiles.loc[0.1]),
        q90=float(quantiles.loc[0.9]),
        min=float(sigma.min()),
        max=float(sigma.max()),
    )


def write_sigma_stats(
    fit_report_path: str | Path,
    *,
    output_dir: str | Path,
    line_ids: Iterable[str] | None = None,
    clean_lines_name: str = "sigma_clean_good_lines.csv",
    summary_name: str = "sigma_clean_good_summary.csv",
) -> tuple[Path, Path, SigmaStats]:
    """Read a fit report, write selected clean lines and sigma summary CSVs."""
    fit_table = pd.read_csv(fit_report_path)
    mask = clean_sigma_mask(fit_table, line_ids=line_ids)
    clean = fit_table.loc[mask].copy()
    stats = summarize_sigma(fit_table, mask)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean_path = out / clean_lines_name
    summary_path = out / summary_name
    clean.to_csv(clean_path, index=False)
    pd.DataFrame([stats.as_dict()]).to_csv(summary_path, index=False)
    return clean_path, summary_path, stats


def _string_column(fit_table: pd.DataFrame, column: str) -> pd.Series:
    if column not in fit_table:
        return pd.Series([""] * len(fit_table), index=fit_table.index, dtype=str)
    return fit_table[column].fillna("").astype(str)


def _bool_column(fit_table: pd.DataFrame, column: str) -> pd.Series:
    if column not in fit_table:
        return pd.Series([False] * len(fit_table), index=fit_table.index, dtype=bool)
    values = fit_table[column]
    if values.dtype == bool:
        return values
    return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
