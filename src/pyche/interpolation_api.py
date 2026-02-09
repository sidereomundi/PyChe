"""Interpolation runtime API used by the engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np

from .interpolation_compat import CompatInterpolator
from .model_tables import ModelTables


class InterpolationKernel(Protocol):
    def interp(self, mass: float, zeta: float, binmax: float) -> Tuple[np.ndarray, float]:
        ...


@dataclass
class ModelInterpolator:
    """Engine-facing interpolation adapter backed by compatibility kernels."""

    tables: ModelTables

    def __post_init__(self) -> None:
        self._compat = CompatInterpolator(self.tables)

    def interp(self, mass: float, zeta: float, binmax: float) -> Tuple[np.ndarray, float]:
        return self._compat.interp(mass, zeta, binmax)
