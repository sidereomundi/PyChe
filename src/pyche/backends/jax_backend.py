"""JAX-backed interpolation backend with batched cache interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..config import RunConfig
from ..constants import GALACTIC_AGE
from ..interpolation_compat import CompatInterpolator
from ..model_tables import ModelTables

try:
    import jax
    from jax import config as jax_config
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - optional dependency
    jax = None
    jax_config = None
    jnp = None
    _JAX_IMPORT_ERROR = exc
else:
    _JAX_IMPORT_ERROR = None
    # Match NumPy/Cython float64 behavior used throughout the model.
    jax_config.update("jax_enable_x64", True)


if jax is not None:

    @jax.jit
    def _interp_mass_zeta(
        x: jnp.ndarray,
        y: jnp.ndarray,
        x_grid: jnp.ndarray,
        y_grid: jnp.ndarray,
        q_grid: jnp.ndarray,
        h_grid: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        ix = jnp.clip(jnp.searchsorted(x_grid, x, side="right") - 1, 0, x_grid.shape[0] - 2)
        iy = jnp.clip(jnp.searchsorted(y_grid, y, side="right") - 1, 0, y_grid.shape[0] - 2)

        x0 = x_grid[ix]
        x1 = x_grid[ix + 1]
        y0 = y_grid[iy]
        y1 = y_grid[iy + 1]
        tx = jnp.where(x1 == x0, 0.0, (x - x0) / (x1 - x0))
        ty = jnp.where(y1 == y0, 0.0, (y - y0) / (y1 - y0))

        q00 = q_grid[ix, iy, :]
        q10 = q_grid[ix + 1, iy, :]
        q01 = q_grid[ix, iy + 1, :]
        q11 = q_grid[ix + 1, iy + 1, :]
        h00 = h_grid[ix, iy]
        h10 = h_grid[ix + 1, iy]
        h01 = h_grid[ix, iy + 1]
        h11 = h_grid[ix + 1, iy + 1]

        q0 = q00 * (1.0 - ty)[:, None] + q01 * ty[:, None]
        q1 = q10 * (1.0 - ty)[:, None] + q11 * ty[:, None]
        q = q0 * (1.0 - tx)[:, None] + q1 * tx[:, None]

        h0 = h00 * (1.0 - ty) + h01 * ty
        h1 = h10 * (1.0 - ty) + h11 * ty
        h = h0 * (1.0 - tx) + h1 * tx
        return q, h

    @jax.jit
    def _interp_batch_cached(
        mass: jnp.ndarray,
        zeta: float,
        binmax: jnp.ndarray,
        zeta_grid: jnp.ndarray,
        mass_grid: jnp.ndarray,
        binmax_grid: jnp.ndarray,
        zero_q: jnp.ndarray,
        zero_h: jnp.ndarray,
        pos_q_base: jnp.ndarray,
        pos_h_base: jnp.ndarray,
        qia_vec: jnp.ndarray,
        qia_sum: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        zeta_arr = jnp.full_like(mass, zeta, dtype=jnp.float64)
        q_zero, h_zero = _interp_mass_zeta(mass, zeta_arr, mass_grid, zeta_grid, zero_q, zero_h)
        q_pos_base, h_pos_base = _interp_mass_zeta(binmax, zeta_arr, binmax_grid, zeta_grid, pos_q_base, pos_h_base)

        ratio = jnp.where(binmax > 0.0, (mass + binmax) / jnp.maximum(binmax, 1.0e-30), 0.0)
        q_pos = q_pos_base + ratio[:, None] * qia_vec[None, :]
        h_pos = h_pos_base + ratio * qia_sum

        # Negative-bin branch matches CompatInterpolator.interp for binmax < 0.
        q_neg = jnp.zeros_like(q_zero)
        cond_li = (binmax < 0.0) & (binmax >= -8.0)
        cond_r = binmax < -8.0
        # q[31] in Fortran-style 1-based indexing maps to index 30 here.
        q_neg = q_neg.at[:, 30].set(jnp.where(cond_li, 8.0e-6, 0.0))
        # q[24:31] -> indices 23..29
        value1 = 0.8e-6 * 20.0
        q24 = value1 * 0.136
        q25 = value1
        q26 = value1 * (0.117 * 151.0 / 138.0)
        q27 = value1 * (3.16 * 88.0 / 138.0)
        q28 = value1 * (1.625 * 89.0 / 138.0 / 3.0)
        q29 = value1 * (2.53 * 90.0 / 138.0)
        q30 = value1 * (3.16 * 86.0 / 138.0)
        q_neg = q_neg.at[:, 23].set(jnp.where(cond_r, q24, q_neg[:, 23]))
        q_neg = q_neg.at[:, 24].set(jnp.where(cond_r, q25, q_neg[:, 24]))
        q_neg = q_neg.at[:, 25].set(jnp.where(cond_r, q26, q_neg[:, 25]))
        q_neg = q_neg.at[:, 26].set(jnp.where(cond_r, q27, q_neg[:, 26]))
        q_neg = q_neg.at[:, 27].set(jnp.where(cond_r, q28, q_neg[:, 27]))
        q_neg = q_neg.at[:, 28].set(jnp.where(cond_r, q29, q_neg[:, 28]))
        q_neg = q_neg.at[:, 29].set(jnp.where(cond_r, q30, q_neg[:, 29]))
        h_neg = jnp.sum(q_neg[:, :31], axis=1)

        zeta_in = (zeta_arr >= zeta_grid[0]) & (zeta_arr <= zeta_grid[-1])
        mass_in = (mass >= mass_grid[0]) & (mass <= mass_grid[-1])
        bin_in = (binmax >= binmax_grid[0]) & (binmax <= binmax_grid[-1])
        use_zero = (binmax == 0.0) & zeta_in & mass_in
        use_pos = (binmax > 0.0) & zeta_in & bin_in
        use_neg = binmax < 0.0
        used = use_zero | use_pos | use_neg

        q = jnp.where(use_zero[:, None], q_zero, jnp.zeros_like(q_zero))
        h = jnp.where(use_zero, h_zero, jnp.zeros_like(h_zero))
        q = jnp.where(use_pos[:, None], q_pos, q)
        h = jnp.where(use_pos, h_pos, h)
        q = jnp.where(use_neg[:, None], q_neg, q)
        h = jnp.where(use_neg, h_neg, h)
        return q, h, used


class JaxCompatInterpolator(CompatInterpolator):
    """Compat interpolator with JAX math primitives for interpolation helpers."""

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


@dataclass
class JaxModelInterpolator:
    """Engine-facing interpolation adapter with batched JAX cache interpolation."""

    tables: ModelTables
    enable_lookup_cache: bool = False
    cache_mass_points: int = 96
    cache_zeta_points: int = 64
    cache_binmax_points: int = 64
    cache_zeta_max: float = 0.05

    def __post_init__(self) -> None:
        self._compat = JaxCompatInterpolator(self.tables)
        self._cache_ready = False
        self._profile = {
            "cache_build_s": 0.0,
            "interp_calls": 0,
            "interp_total_s": 0.0,
            "interp_cached_hits": 0,
            "interp_fallback_hits": 0,
            "interp_many_calls": 0,
            "interp_many_total_s": 0.0,
            "interp_many_cached_batch_s": 0.0,
            "interp_many_fallback_exact_s": 0.0,
            "interp_many_indices_total": 0,
            "interp_many_fallback_points": 0,
        }

        self._qia_vec = np.zeros(33, dtype=np.float64)
        self._qia_vec[0:23] = np.array(
            [
                0.0,
                0.048,
                0.143,
                1.16e-6,
                1.40e-6,
                0.00202,
                0.0425 * 1.2,
                0.154,
                0.6,
                0.0,
                0.0,
                0.0846,
                0.0119,
                0.0,
                6.345e-4,
                1.19e-3,
                1.28e-5,
                4.16e-4,
                7.49e-5,
                5.67e-3,
                6.21e-3,
                1.56e-3,
                1.83e-2,
            ],
            dtype=np.float64,
        )
        self._qia_sum = float(np.sum(self._qia_vec[:31], dtype=np.float64))
        if self.enable_lookup_cache:
            self._build_lookup_cache()

    def _interp_exact(self, mass: float, zeta: float, binmax: float):
        return self._compat.interp(mass, zeta, binmax)

    def _build_lookup_cache(self) -> None:
        t0 = perf_counter()
        zeta_max = float(self.cache_zeta_max)
        self._zeta_grid = np.linspace(0.0, zeta_max, int(self.cache_zeta_points), dtype=np.float64)

        mass_samples = np.asarray(self.tables.massa[1 : self.tables.ninputyield + 1], dtype=np.float64)
        mass_samples = mass_samples[np.isfinite(mass_samples)]
        mass_samples = mass_samples[mass_samples > 0.0]
        mass_min = float(np.min(mass_samples)) if mass_samples.size else 0.1
        mass_max = float(np.max(mass_samples)) if mass_samples.size else 100.0
        mass_min = min(mass_min, 0.12)
        mass_max = max(mass_max, 100.0)
        self._mass_grid = np.linspace(mass_min, mass_max, int(self.cache_mass_points), dtype=np.float64)
        self._binmax_grid = np.linspace(0.5, 8.0, int(self.cache_binmax_points), dtype=np.float64)

        nz = self._zeta_grid.size
        nm = self._mass_grid.size
        nb = self._binmax_grid.size

        self._cache_zero_q = np.empty((nm, nz, 33), dtype=np.float64)
        self._cache_zero_h = np.empty((nm, nz), dtype=np.float64)
        self._cache_pos_q_base = np.empty((nb, nz, 33), dtype=np.float64)
        self._cache_pos_h_base = np.empty((nb, nz), dtype=np.float64)

        for i, mass in enumerate(self._mass_grid):
            for j, zeta in enumerate(self._zeta_grid):
                q, h = self._interp_exact(float(mass), float(zeta), 0.0)
                self._cache_zero_q[i, j, :] = q
                self._cache_zero_h[i, j] = float(h)

        qia_ref = self._qia_vec[:31] * 2.0
        hia_ref = self._qia_sum * 2.0
        for i, binmax in enumerate(self._binmax_grid):
            for j, zeta in enumerate(self._zeta_grid):
                q, h = self._interp_exact(float(binmax), float(zeta), float(binmax))
                q_base = np.asarray(q, dtype=np.float64).copy()
                q_base[:31] -= qia_ref
                self._cache_pos_q_base[i, j, :] = q_base
                self._cache_pos_h_base[i, j] = float(h) - hia_ref

        self._zeta_grid_j = jnp.asarray(self._zeta_grid, dtype=jnp.float64)
        self._mass_grid_j = jnp.asarray(self._mass_grid, dtype=jnp.float64)
        self._binmax_grid_j = jnp.asarray(self._binmax_grid, dtype=jnp.float64)
        self._cache_zero_q_j = jnp.asarray(self._cache_zero_q, dtype=jnp.float64)
        self._cache_zero_h_j = jnp.asarray(self._cache_zero_h, dtype=jnp.float64)
        self._cache_pos_q_base_j = jnp.asarray(self._cache_pos_q_base, dtype=jnp.float64)
        self._cache_pos_h_base_j = jnp.asarray(self._cache_pos_h_base, dtype=jnp.float64)
        self._qia_vec_j = jnp.asarray(self._qia_vec, dtype=jnp.float64)
        self._cache_ready = True
        self._profile["cache_build_s"] += perf_counter() - t0

    def interp(self, mass: float, zeta: float, binmax: float):
        t0 = perf_counter()
        self._profile["interp_calls"] += 1
        if not self._cache_ready:
            out = self._interp_exact(mass, zeta, binmax)
            self._profile["interp_fallback_hits"] += 1
            self._profile["interp_total_s"] += perf_counter() - t0
            return out
        qj, hj, usedj = _interp_batch_cached(
            jnp.asarray([mass], dtype=jnp.float64),
            float(zeta),
            jnp.asarray([binmax], dtype=jnp.float64),
            self._zeta_grid_j,
            self._mass_grid_j,
            self._binmax_grid_j,
            self._cache_zero_q_j,
            self._cache_zero_h_j,
            self._cache_pos_q_base_j,
            self._cache_pos_h_base_j,
            self._qia_vec_j,
            float(self._qia_sum),
        )
        if bool(np.asarray(usedj)[0]):
            self._profile["interp_cached_hits"] += 1
            self._profile["interp_total_s"] += perf_counter() - t0
            return np.asarray(qj[0], dtype=np.float64), float(np.asarray(hj)[0])
        out = self._interp_exact(mass, zeta, binmax)
        self._profile["interp_fallback_hits"] += 1
        self._profile["interp_total_s"] += perf_counter() - t0
        return out

    def interp_many_step(
        self,
        *,
        t: int,
        sfr_mass: float,
        zeta_t: float,
        elem: int,
        indices: np.ndarray,
        mstars: np.ndarray,
        binmax: np.ndarray,
        multi1: np.ndarray,
        tdead: np.ndarray,
        hecores: np.ndarray,
        mstars1_eff: np.ndarray,
        qispecial: np.ndarray,
    ) -> float:
        t0_total = perf_counter()
        self._profile["interp_many_calls"] += 1
        idx = np.asarray(indices, dtype=np.int32)
        if idx.size == 0:
            self._profile["interp_many_total_s"] += perf_counter() - t0_total
            return 0.0
        self._profile["interp_many_indices_total"] += int(idx.size)

        old_mask = (np.asarray(tdead[idx], dtype=np.float64) + float(t)) > GALACTIC_AGE
        oldstars_contrib = float(np.sum(np.asarray(multi1[idx], dtype=np.float64)[old_mask], dtype=np.float64) * sfr_mass)

        mloc = np.asarray(mstars[idx], dtype=np.float64)
        bloc = np.asarray(binmax[idx], dtype=np.float64)
        mstars1_eff[idx] = np.where(bloc > 0.0, bloc + mloc, mloc)

        if self._cache_ready:
            t0_cached = perf_counter()
            qj, hj, usedj = _interp_batch_cached(
                jnp.asarray(mloc, dtype=jnp.float64),
                float(zeta_t),
                jnp.asarray(bloc, dtype=jnp.float64),
                self._zeta_grid_j,
                self._mass_grid_j,
                self._binmax_grid_j,
                self._cache_zero_q_j,
                self._cache_zero_h_j,
                self._cache_pos_q_base_j,
                self._cache_pos_h_base_j,
                self._qia_vec_j,
                float(self._qia_sum),
            )
            self._profile["interp_many_cached_batch_s"] += perf_counter() - t0_cached
            q_np = np.asarray(qj, dtype=np.float64)
            h_np = np.asarray(hj, dtype=np.float64)
            used = np.asarray(usedj, dtype=bool)

            if np.any(used):
                used_idx = idx[used]
                hecores[used_idx] = h_np[used]
                qispecial[1:elem, used_idx] = q_np[used, : elem - 1].T

            unresolved = idx[~used]
            self._profile["interp_many_fallback_points"] += int(unresolved.size)
            t0_fallback = perf_counter()
            for jj in unresolved:
                q, h = self._interp_exact(float(mstars[jj]), float(zeta_t), float(binmax[jj]))
                hecores[jj] = h
                qispecial[1:elem, jj] = q[: elem - 1]
            self._profile["interp_many_fallback_exact_s"] += perf_counter() - t0_fallback
            self._profile["interp_many_total_s"] += perf_counter() - t0_total
            return oldstars_contrib

        self._profile["interp_many_fallback_points"] += int(idx.size)
        t0_fallback = perf_counter()
        for jj in idx:
            q, h = self._interp_exact(float(mstars[jj]), float(zeta_t), float(binmax[jj]))
            hecores[jj] = h
            qispecial[1:elem, jj] = q[: elem - 1]
        self._profile["interp_many_fallback_exact_s"] += perf_counter() - t0_fallback
        self._profile["interp_many_total_s"] += perf_counter() - t0_total
        return oldstars_contrib

    def profile_snapshot(self) -> dict[str, float | int]:
        return dict(self._profile)


def build_jax_backend(tables: ModelTables, cfg: RunConfig | None = None):
    if jnp is None:
        raise RuntimeError(f"JAX backend unavailable: {_JAX_IMPORT_ERROR}") from _JAX_IMPORT_ERROR
    if cfg is None:
        return JaxModelInterpolator(tables, enable_lookup_cache=False)
    return JaxModelInterpolator(
        tables,
        enable_lookup_cache=bool(cfg.interp_cache),
        cache_mass_points=int(cfg.interp_cache_mass_points),
        cache_zeta_points=int(cfg.interp_cache_zeta_points),
        cache_binmax_points=int(cfg.interp_cache_binmax_points),
        cache_zeta_max=float(cfg.interp_cache_zeta_max),
    )
