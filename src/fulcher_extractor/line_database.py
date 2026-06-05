"""Fulcher-alpha transition database loading."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback.
    import tomli as tomllib  # type: ignore[no-redef]

import pandas as pd

RESOURCE_PACKAGE = "fulcher_extractor.resources"
DEFAULT_LINE_RESOURCE = "fulcher_alpha_lines.toml"


@dataclass(frozen=True)
class FulcherLine:
    """One Fulcher-alpha transition in normalized wavelength units."""

    isotopologue: str
    branch: str
    v_upper: int
    v_lower: int
    band: str
    N: int
    wavelength_nm: float
    source_table: str
    original_unit: str
    status: str = "active"

    @property
    def line_id(self) -> str:
        return f"{self.isotopologue}_{self.branch}{self.N}_{self.band}"


def _resource_bytes(resource_name: str = DEFAULT_LINE_RESOURCE) -> bytes:
    return files(RESOURCE_PACKAGE).joinpath(resource_name).read_bytes()


def load_lines(path: str | Path | None = None) -> list[FulcherLine]:
    """Load Fulcher-alpha lines from a TOML resource or explicit path."""
    if path is None:
        payload = tomllib.loads(_resource_bytes().decode("utf-8"))
    else:
        with Path(path).open("rb") as f:
            payload = tomllib.load(f)

    lines = []
    for row in payload.get("lines", []):
        if float(row["wavelength_nm"]) <= 0:
            continue
        lines.append(
            FulcherLine(
                isotopologue=str(row["isotopologue"]),
                branch=str(row["branch"]),
                v_upper=int(row["v_upper"]),
                v_lower=int(row["v_lower"]),
                band=str(row["band"]),
                N=int(row["N"]),
                wavelength_nm=float(row["wavelength_nm"]),
                source_table=str(row.get("source_table", "")),
                original_unit=str(row.get("original_unit", "nm")),
                status=str(row.get("status", "active")),
            )
        )
    return lines


def filter_lines(
    lines: Iterable[FulcherLine],
    *,
    isotopologue: str = "H2",
    branch: str | None = "Q",
    bands: Iterable[str] | None = None,
    n_min: int | None = None,
    n_max: int | None = None,
    wavelength_min_nm: float | None = None,
    wavelength_max_nm: float | None = None,
) -> list[FulcherLine]:
    """Return lines matching common extraction filters."""
    band_set = set(bands) if bands is not None else None
    selected = []
    for line in lines:
        if line.isotopologue != isotopologue:
            continue
        if branch is not None and line.branch != branch:
            continue
        if band_set is not None and line.band not in band_set:
            continue
        if n_min is not None and line.N < n_min:
            continue
        if n_max is not None and line.N > n_max:
            continue
        if wavelength_min_nm is not None and line.wavelength_nm < wavelength_min_nm:
            continue
        if wavelength_max_nm is not None and line.wavelength_nm > wavelength_max_nm:
            continue
        selected.append(line)
    return selected


def lines_to_dataframe(lines: Iterable[FulcherLine]) -> pd.DataFrame:
    """Convert loaded lines to a tabular representation."""
    return pd.DataFrame([line.__dict__ | {"line_id": line.line_id} for line in lines])


def wavelength_matrix(
    lines: Iterable[FulcherLine], *, isotopologue: str = "H2"
) -> pd.DataFrame:
    """Return a fulcheranalyzer-oriented wavelength matrix indexed by N."""
    df = lines_to_dataframe(filter_lines(lines, isotopologue=isotopologue))
    if df.empty:
        return pd.DataFrame()
    matrix = df.pivot(index="N", columns="band", values="wavelength_nm")
    bands = sorted(matrix.columns, key=lambda label: int(str(label).split("-")[0]))
    return matrix.reindex(columns=bands).sort_index()
