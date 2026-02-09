"""Lightweight interpolation utilities kept for backward compatibility tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class InterpolationData:
    massa: np.ndarray = field(default_factory=lambda: np.empty(0))
    zeta: np.ndarray = field(default_factory=lambda: np.empty(0))
    W: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 0)))


@dataclass
class Interpolator:
    """Simple bilinear interpolator used by unit tests."""

    data: InterpolationData

    def interp(self, mass: float, zeta: float, binmax: float) -> Tuple[np.ndarray, float]:
        masses = self.data.massa
        zetas = self.data.zeta
        W = self.data.W

        i = np.searchsorted(masses, mass) - 1
        j = np.searchsorted(zetas, zeta) - 1
        i = np.clip(i, 0, len(masses) - 2)
        j = np.clip(j, 0, len(zetas) - 2)

        m1, m2 = masses[i], masses[i + 1]
        z1, z2 = zetas[j], zetas[j + 1]
        fm = 0.0 if m2 == m1 else (mass - m1) / (m2 - m1)
        fz = 0.0 if z2 == z1 else (zeta - z1) / (z2 - z1)

        q = (
            W[:, i, j] * (1 - fm) * (1 - fz)
            + W[:, i + 1, j] * fm * (1 - fz)
            + W[:, i, j + 1] * (1 - fm) * fz
            + W[:, i + 1, j + 1] * fm * fz
        )
        return q, 0.1 * mass
