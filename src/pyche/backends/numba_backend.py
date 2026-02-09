"""Numba-accelerated interpolation backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interpolation_compat import CompatInterpolator
from ..model_tables import ModelTables

try:
    from numba import njit
except Exception as exc:  # pragma: no cover - optional dependency
    njit = None
    _NUMBA_IMPORT_ERROR = exc
else:
    _NUMBA_IMPORT_ERROR = None


if njit is not None:

    @njit(cache=True)
    def _polint_numba(xa: np.ndarray, ya: np.ndarray, x: float) -> tuple[float, float]:
        n = len(xa)
        c = ya.astype(np.float64).copy()
        d = ya.astype(np.float64).copy()
        ns = 0
        best = abs(x - xa[0])
        for i in range(1, n):
            diff = abs(x - xa[i])
            if diff < best:
                best = diff
                ns = i
        y = float(ya[ns])
        ns -= 1
        dy = 0.0
        for m in range(1, n):
            for i in range(n - m):
                ho = xa[i] - x
                hp = xa[i + m] - x
                w = c[i + 1] - d[i]
                den = ho - hp
                if den == 0.0:
                    return y, dy
                den = w / den
                d[i] = hp * den
                c[i] = ho * den
            if 2 * (ns + 1) < n - m:
                dy = float(c[ns + 1])
            else:
                dy = float(d[ns])
                ns -= 1
            y += dy
        return float(y), float(dy)

    # Prime JIT cache once to avoid a latency spike in the first hot iteration.
    _polint_numba(np.array([0.0, 1.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64), 0.5)


class NumbaCompatInterpolator(CompatInterpolator):
    """Compat interpolator with numba-jitted polynomial interpolation."""

    def polint(self, xa: np.ndarray, ya: np.ndarray, x: float) -> tuple[float, float]:
        if njit is None:
            return super().polint(xa, ya, x)
        return _polint_numba(np.asarray(xa, dtype=np.float64), np.asarray(ya, dtype=np.float64), float(x))


@dataclass
class NumbaModelInterpolator:
    """Engine-facing interpolation adapter backed by numba primitives."""

    tables: ModelTables

    def __post_init__(self) -> None:
        self._compat = NumbaCompatInterpolator(self.tables)

    def interp(self, mass: float, zeta: float, binmax: float):
        return self._compat.interp(mass, zeta, binmax)


def build_numba_backend(tables: ModelTables):
    if njit is None:
        raise RuntimeError(
            f"Numba backend unavailable: {_NUMBA_IMPORT_ERROR}"
        ) from _NUMBA_IMPORT_ERROR
    return NumbaModelInterpolator(tables)
