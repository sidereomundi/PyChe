"""JAX-backed interpolation backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interpolation_compat import CompatInterpolator
from ..model_tables import ModelTables

try:
    from jax import config as jax_config
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - optional dependency
    jax_config = None
    jnp = None
    _JAX_IMPORT_ERROR = exc
else:
    _JAX_IMPORT_ERROR = None
    # Match NumPy/Cython float64 behavior used throughout the model.
    jax_config.update("jax_enable_x64", True)


class JaxCompatInterpolator(CompatInterpolator):
    """Compat interpolator with JAX math primitives for interpolation steps."""

    @staticmethod
    def _lin2(x0: float, x1: float, y0: float, y1: float, x: float) -> float:
        if jnp is None:
            return CompatInterpolator._lin2(x0, x1, y0, y1, x)
        x0j = jnp.asarray(x0, dtype=jnp.float64)
        x1j = jnp.asarray(x1, dtype=jnp.float64)
        y0j = jnp.asarray(y0, dtype=jnp.float64)
        y1j = jnp.asarray(y1, dtype=jnp.float64)
        xj = jnp.asarray(x, dtype=jnp.float64)
        den = x1j - x0j
        out = jnp.where(den == 0.0, y0j, y0j + (y1j - y0j) * ((xj - x0j) / den))
        return float(out)

    def polint(self, xa: np.ndarray, ya: np.ndarray, x: float) -> tuple[float, float]:
        if jnp is None:
            return super().polint(xa, ya, x)
        xa_j = jnp.asarray(xa, dtype=jnp.float64)
        ya_j = jnp.asarray(ya, dtype=jnp.float64)
        x_j = jnp.asarray(x, dtype=jnp.float64)

        n = int(xa_j.shape[0])
        c = ya_j.copy()
        d = ya_j.copy()
        ns = int(jnp.argmin(jnp.abs(x_j - xa_j)))
        y = float(ya_j[ns])
        ns -= 1
        dy = 0.0

        for m in range(1, n):
            for i in range(n - m):
                ho = xa_j[i] - x_j
                hp = xa_j[i + m] - x_j
                w = c[i + 1] - d[i]
                den = ho - hp
                den_f = float(den)
                if den_f == 0.0:
                    return y, dy
                den2 = w / den
                d = d.at[i].set(hp * den2)
                c = c.at[i].set(ho * den2)
            if 2 * (ns + 1) < n - m:
                dy = float(c[ns + 1])
            else:
                dy = float(d[ns])
                ns -= 1
            y += dy
        return float(y), float(dy)


@dataclass
class JaxModelInterpolator:
    """Engine-facing interpolation adapter backed by JAX primitives."""

    tables: ModelTables

    def __post_init__(self) -> None:
        self._compat = JaxCompatInterpolator(self.tables)

    def interp(self, mass: float, zeta: float, binmax: float):
        return self._compat.interp(mass, zeta, binmax)


def build_jax_backend(tables: ModelTables):
    if jnp is None:
        raise RuntimeError(f"JAX backend unavailable: {_JAX_IMPORT_ERROR}") from _JAX_IMPORT_ERROR
    return JaxModelInterpolator(tables)
