"""Output writers for fulcheranalyzer-compatible intensity tables."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np

from .fit import LineFitResult


def results_to_dataframe(results: Iterable[LineFitResult]) -> pd.DataFrame:
    """Return one long-form audit row per fitted line."""
    rows = []
    for result in results:
        row = asdict(result)
        if not np.isfinite(float(row.get("matrix_amplitude", float("nan")))):
            row["matrix_amplitude"] = row["amplitude"]
        if not np.isfinite(float(row.get("matrix_amplitude_stderr", float("nan")))):
            row["matrix_amplitude_stderr"] = row["amplitude_stderr"]
        rows.append(row)
    return pd.DataFrame(rows)


def results_to_matrices(
    results: Iterable[LineFitResult],
    *,
    value_column: str = "matrix_amplitude",
    error_column: str = "matrix_amplitude_stderr",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fulcheranalyzer-compatible intensity/error matrices."""
    df = results_to_dataframe(results)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    intensity = df.pivot(index="N", columns="band", values=value_column)
    error = df.pivot(index="N", columns="band", values=error_column)
    ordered_bands = sorted(intensity.columns, key=lambda label: int(str(label).split("-")[0]))
    max_n = int(df["N"].max())
    index = list(range(1, max_n + 1))
    intensity = intensity.reindex(index=index, columns=ordered_bands).fillna(0.0)
    error = error.reindex(index=index, columns=ordered_bands).fillna(0.0)
    return intensity, error


def write_fulcheranalyzer_csvs(
    results: Iterable[LineFitResult],
    *,
    output_dir: str | Path,
    shot: str | int,
    frame: str | int,
    gas: str = "H2",
    metadata: dict[str, object] | None = None,
) -> tuple[Path, Path, Path]:
    """Write intensity, error, and long-form fit report CSVs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    materialized = list(results)
    intensity, error = results_to_matrices(materialized)
    fit_table = results_to_dataframe(materialized)
    stem = f"{shot}_fr_{frame}"
    intensity_path = out / f"{stem}.csv"
    error_path = out / f"{stem}_err.csv"
    fit_path = out / f"{stem}_fit_report.csv"
    header = _metadata_header(shot=shot, frame=frame, gas=gas, metadata=metadata or {})
    intensity.to_csv(intensity_path, index=False, header=False)
    error.to_csv(error_path, index=False, header=False)
    fit_table.to_csv(fit_path, index=False)
    for path in (intensity_path, error_path):
        path.write_text(header + path.read_text())
    return intensity_path, error_path, fit_path


def _metadata_header(
    *, shot: str | int, frame: str | int, gas: str, metadata: dict[str, object]
) -> str:
    lines = [
        f"# shotnumber: {shot}",
        f"# frame : {frame}",
        f"# gas : {gas}",
        "# Columns: diagonal Fulcher band index",
        "# Rows: rotational quantum number N",
        "# Values: fitted Gaussian area/intensity",
    ]
    for key, value in metadata.items():
        lines.append(f"# {key}: {value}")
    lines.append("# [Data]")
    return "\n".join(lines) + "\n"
