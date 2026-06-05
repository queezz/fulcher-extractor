"""SpectroCube loading and wavelength-window selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class Spectrum:
    """A single 1D spectrum selected from a SpectroCube."""

    source_path: Path
    shot_id: str | None
    selectors: dict[str, Any]
    wavelength_nm: np.ndarray
    intensity: np.ndarray
    intensity_units: str
    wavelength_medium: str
    metadata: dict[str, Any]

    def window(self, wavelength_min_nm: float, wavelength_max_nm: float) -> "SpectrumWindow":
        mask = (self.wavelength_nm >= wavelength_min_nm) & (
            self.wavelength_nm <= wavelength_max_nm
        )
        return SpectrumWindow(
            wavelength_nm=self.wavelength_nm[mask],
            intensity=self.intensity[mask],
            wavelength_min_nm=wavelength_min_nm,
            wavelength_max_nm=wavelength_max_nm,
        )


@dataclass(frozen=True)
class SpectrumWindow:
    wavelength_nm: np.ndarray
    intensity: np.ndarray
    wavelength_min_nm: float
    wavelength_max_nm: float


def parse_shot_id(path: str | Path, attrs: dict[str, Any] | None = None) -> str | None:
    """Parse shot id from SpectroCube metadata or filename."""
    attrs = attrs or {}
    for key in ("shot_number", "shot_id"):
        if key in attrs and attrs[key] not in (None, ""):
            return str(attrs[key])
    match = re.match(r"(\d+)", Path(path).name)
    return match.group(1) if match else None


def open_spectrocube(path: str | Path, *, engine: str | None = None) -> xr.Dataset:
    """Open a SpectroCube dataset eagerly and validate required fields."""
    try:
        ds = xr.load_dataset(path, engine=engine)
    except ValueError as exc:
        if "IO backends" in str(exc) or "found the following matches" in str(exc):
            raise RuntimeError(
                "xarray could not open this NetCDF4 SpectroCube. Install a backend "
                "such as h5netcdf or netCDF4 in the active environment."
            ) from exc
        raise
    if "wavelength" not in ds.coords:
        raise ValueError("SpectroCube is missing required 'wavelength' coordinate.")
    if "intensity" not in ds.data_vars:
        raise ValueError("SpectroCube is missing required 'intensity' data variable.")
    if "wavelength" not in ds["intensity"].dims:
        raise ValueError("SpectroCube 'intensity' must depend on 'wavelength'.")
    return ds


def load_spectrum(
    path: str | Path,
    *,
    frame: int | float | None = None,
    selectors: dict[str, Any] | None = None,
    engine: str | None = None,
) -> Spectrum:
    """Load one 1D spectrum from a SpectroCube.

    Non-wavelength dimensions must be reduced to scalar selections. If a
    ``frame`` dimension is present and no frame is supplied, frame 0 is used.
    """
    source_path = Path(path)
    ds = open_spectrocube(source_path, engine=engine)
    intensity = ds["intensity"]
    resolved_selectors = dict(selectors or {})
    non_wavelength_dims = [dim for dim in intensity.dims if dim != "wavelength"]

    if "frame" in non_wavelength_dims and "frame" not in resolved_selectors:
        resolved_selectors["frame"] = 0 if frame is None else frame
    elif frame is not None:
        resolved_selectors["frame"] = frame

    for dim in non_wavelength_dims:
        if dim not in resolved_selectors:
            raise ValueError(
                f"Selection required for non-wavelength SpectroCube dimension {dim!r}."
            )

    selected = intensity.sel(resolved_selectors)
    if selected.ndim != 1:
        raise ValueError(f"Selected intensity must be 1D, got dims {selected.dims!r}.")

    return Spectrum(
        source_path=source_path,
        shot_id=parse_shot_id(source_path, ds.attrs),
        selectors=resolved_selectors,
        wavelength_nm=np.asarray(ds["wavelength"].values, dtype=float),
        intensity=np.asarray(selected.values, dtype=float),
        intensity_units=str(ds.attrs.get("intensity_units", selected.attrs.get("units", ""))),
        wavelength_medium=str(ds.attrs.get("wavelength_medium", "")),
        metadata=dict(ds.attrs),
    )
