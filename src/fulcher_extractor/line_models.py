"""Gaussian, blended-Gaussian, and baseline model definitions."""

from __future__ import annotations

import numpy as np

SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def gaussian_area_model(
    wavelength_nm: np.ndarray, area: float, center_nm: float, sigma_nm: float
) -> np.ndarray:
    """Gaussian parameterized by integrated area, matching lmfit amplitude."""
    sigma = max(float(sigma_nm), np.finfo(float).eps)
    return area / (sigma * SQRT_2PI) * np.exp(
        -0.5 * ((wavelength_nm - center_nm) / sigma) ** 2
    )


def local_minimum_baseline(intensity: np.ndarray) -> float:
    """Mechanical baseline used by old manual reductions: subtract local minimum."""
    finite = np.asarray(intensity, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(finite.min())
