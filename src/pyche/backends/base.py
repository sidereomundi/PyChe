"""Backend protocol for interpolation/evolution kernels."""

from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np


class InterpolationBackend(Protocol):
    def interp(self, mass: float, zeta: float, binmax: float) -> Tuple[np.ndarray, float]:
        ...

