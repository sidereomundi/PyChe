"""Cython-accelerated interpolation backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import RunConfig
from ..interpolation_compat import CompatInterpolator
from ..model_tables import ModelTables

try:
    from .. import _cyinterp  # type: ignore
except Exception as exc:  # pragma: no cover - optional compiled extension
    _cyinterp = None
    _CYTHON_IMPORT_ERROR = exc
else:
    _CYTHON_IMPORT_ERROR = None


class CythonCompatInterpolator(CompatInterpolator):
    """Compat interpolator with Cython-accelerated polynomial interpolation."""

    def polint(self, xa: np.ndarray, ya: np.ndarray, x: float):
        if _cyinterp is None:
            return super().polint(xa, ya, x)
        return _cyinterp.polint(np.asarray(xa, dtype=np.float64), np.asarray(ya, dtype=np.float64), float(x))


@dataclass
class CythonModelInterpolator:
    """Engine-facing interpolation adapter backed by Cython primitives."""

    tables: ModelTables
    enable_lookup_cache: bool = False
    cache_mass_points: int = 128
    cache_zeta_points: int = 96
    cache_binmax_points: int = 96
    cache_zeta_max: float = 0.05
    enable_cache_guard: bool = False
    cache_guard_tol: float = 0.05
    cache_guard_stride: int = 32
    cache_guard_samples: int = 5

    def __post_init__(self) -> None:
        self._compat = CythonCompatInterpolator(self.tables)
        self._W = np.asarray(self.tables.W, dtype=np.float64)
        self._massa = np.asarray(self.tables.massa, dtype=np.float64)
        self._massac = np.asarray(self.tables.massac, dtype=np.float64)
        self._MBa = np.asarray(self.tables.MBa, dtype=np.float64)
        self._WBa = np.asarray(self.tables.WBa, dtype=np.float64)
        self._WSr = np.asarray(self.tables.WSr, dtype=np.float64)
        self._WY = np.asarray(self.tables.WY, dtype=np.float64)
        self._WLa = np.asarray(self.tables.WLa, dtype=np.float64)
        self._WZr = np.asarray(self.tables.WZr, dtype=np.float64)
        self._WRb = np.asarray(self.tables.WRb, dtype=np.float64)
        self._WEu = np.asarray(self.tables.WEu, dtype=np.float64)
        self._zbario = np.asarray(self.tables.zbario, dtype=np.float64)
        self._massaba = np.asarray(self.tables.massaba, dtype=np.float64)
        self._ba = np.asarray(self.tables.ba, dtype=np.float64)
        self._sr = np.asarray(self.tables.sr, dtype=np.float64)
        self._yt = np.asarray(self.tables.yt, dtype=np.float64)
        self._eu = np.asarray(self.tables.eu, dtype=np.float64)
        self._zr = np.asarray(self.tables.zr, dtype=np.float64)
        self._la = np.asarray(self.tables.la, dtype=np.float64)
        self._rb = np.asarray(self.tables.rb, dtype=np.float64)
        self._YLi = np.asarray(self.tables.YLi, dtype=np.float64)
        self._massaLi = np.asarray(self.tables.massaLi, dtype=np.float64)
        self._cache_ready = False
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
        self._guard_calls = 0
        if self.enable_lookup_cache and _cyinterp is not None:
            self._build_lookup_cache()

    def _interp_exact(self, mass: float, zeta: float, binmax: float):
        if _cyinterp is None:
            return self._compat.interp(mass, zeta, binmax)
        return _cyinterp.interp_full(
            float(mass),
            float(zeta),
            float(binmax),
            int(self.tables.ninputyield),
            self._W,
            self._massa,
            self._massac,
            self._MBa,
            self._WBa,
            self._WSr,
            self._WY,
            self._WLa,
            self._WZr,
            self._WRb,
            self._WEu,
            self._zbario,
            self._massaba,
            self._ba,
            self._sr,
            self._yt,
            self._eu,
            self._zr,
            self._la,
            self._rb,
            self._YLi,
            self._massaLi,
        )

    def _build_lookup_cache(self) -> None:
        zeta_max = float(self.cache_zeta_max)
        self._zeta_grid = np.linspace(0.0, zeta_max, int(self.cache_zeta_points), dtype=np.float64)

        mass_samples = self._massa[1 : self.tables.ninputyield + 1]
        mass_samples = mass_samples[np.isfinite(mass_samples)]
        mass_samples = mass_samples[mass_samples > 0.0]
        mass_min = float(np.min(mass_samples)) if mass_samples.size else 0.1
        mass_max = float(np.max(mass_samples)) if mass_samples.size else 100.0
        mass_min = min(mass_min, 0.12)
        mass_max = max(mass_max, 100.0)
        self._mass_grid = np.linspace(mass_min, mass_max, int(self.cache_mass_points), dtype=np.float64)

        # Positive-bin branch: H is effectively binmax and ratio is applied analytically.
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

        self._cache_ready = True

    def interp(self, mass: float, zeta: float, binmax: float):
        if not self._cache_ready:
            return self._interp_exact(mass, zeta, binmax)
        q, h, used = _cyinterp.interp_from_cache(
            float(mass),
            float(zeta),
            float(binmax),
            self._zeta_grid,
            self._mass_grid,
            self._binmax_grid,
            self._cache_zero_q,
            self._cache_zero_h,
            self._cache_pos_q_base,
            self._cache_pos_h_base,
            self._qia_vec,
            float(self._qia_sum),
        )
        if bool(used):
            return np.asarray(q, dtype=np.float64), float(h)
        return self._interp_exact(mass, zeta, binmax)

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
        def _use_cached_bulk() -> bool:
            if not self._cache_ready:
                return False
            if not self.enable_cache_guard:
                return True
            self._guard_calls += 1
            if (self._guard_calls % max(1, int(self.cache_guard_stride))) != 0:
                return True
            idxs = np.asarray(indices, dtype=np.int32)
            if idxs.size == 0:
                return True
            n_samples = max(1, int(self.cache_guard_samples))
            pos = np.linspace(0, idxs.size - 1, num=min(n_samples, idxs.size), dtype=np.int32)
            sample_bins = idxs[np.unique(pos)]
            zt = float(zeta_t)
            tol = float(self.cache_guard_tol)
            key_tol = min(tol, 0.02)
            # O16, Mg, Fe in q/qqn indexing
            key_idx = np.array([3, 7, 9], dtype=np.int32)

            for jj in sample_bins:
                out_cached = _cyinterp.interp_from_cache(
                    float(mstars[int(jj)]),
                    zt,
                    float(binmax[int(jj)]),
                    self._zeta_grid,
                    self._mass_grid,
                    self._binmax_grid,
                    self._cache_zero_q,
                    self._cache_zero_h,
                    self._cache_pos_q_base,
                    self._cache_pos_h_base,
                    self._qia_vec,
                    float(self._qia_sum),
                )
                if not bool(out_cached[2]):
                    continue
                q_exact, h_exact = self._interp_exact(float(mstars[int(jj)]), zt, float(binmax[int(jj)]))
                q_cached = np.asarray(out_cached[0], dtype=np.float64)
                h_cached = float(out_cached[1])

                rel_q = float(
                    np.max(
                        np.abs(q_cached[: elem - 1] - q_exact[: elem - 1])
                        / np.maximum(np.abs(q_exact[: elem - 1]), 1.0e-20)
                    )
                )
                kidx = key_idx[key_idx < (elem - 1)]
                rel_key = float(
                    np.max(np.abs(q_cached[kidx] - q_exact[kidx]) / np.maximum(np.abs(q_exact[kidx]), 1.0e-20))
                ) if kidx.size > 0 else 0.0
                rel_h = abs(h_cached - float(h_exact)) / max(abs(float(h_exact)), 1.0e-20)
                if (rel_q > tol) or (rel_key > key_tol) or (rel_h > tol):
                    return False
            return True

        if _cyinterp is None:
            oldstars_contrib = 0.0
            for jj in np.asarray(indices, dtype=np.int32):
                if tdead[jj] + t > 13500.0:
                    oldstars_contrib += multi1[jj] * sfr_mass
                q, hecore = self.interp(float(mstars[jj]), float(zeta_t), float(binmax[jj]))
                mstars1_eff[jj] = (binmax[jj] + mstars[jj]) if (binmax[jj] > 0.0) else mstars[jj]
                hecores[jj] = hecore
                qispecial[1:elem, jj] = q[: elem - 1]
            return float(oldstars_contrib)

        if _use_cached_bulk():
            return float(
                _cyinterp.interp_many_cached(
                    int(t),
                    float(sfr_mass),
                    float(zeta_t),
                    int(elem),
                    np.asarray(indices, dtype=np.int32),
                    np.asarray(mstars, dtype=np.float64),
                    np.asarray(binmax, dtype=np.float64),
                    np.asarray(multi1, dtype=np.float64),
                    np.asarray(tdead, dtype=np.float64),
                    np.asarray(hecores, dtype=np.float64),
                    np.asarray(mstars1_eff, dtype=np.float64),
                    np.asarray(qispecial, dtype=np.float64),
                    self._zeta_grid,
                    self._mass_grid,
                    self._binmax_grid,
                    self._cache_zero_q,
                    self._cache_zero_h,
                    self._cache_pos_q_base,
                    self._cache_pos_h_base,
                    self._qia_vec,
                    float(self._qia_sum),
                    int(self.tables.ninputyield),
                    self._W,
                    self._massa,
                    self._massac,
                    self._MBa,
                    self._WBa,
                    self._WSr,
                    self._WY,
                    self._WLa,
                    self._WZr,
                    self._WRb,
                    self._WEu,
                    self._zbario,
                    self._massaba,
                    self._ba,
                    self._sr,
                    self._yt,
                    self._eu,
                    self._zr,
                    self._la,
                    self._rb,
                    self._YLi,
                    self._massaLi,
                )
            )

        return float(
            _cyinterp.interp_many_full(
                int(t),
                float(sfr_mass),
                float(zeta_t),
                int(elem),
                np.asarray(indices, dtype=np.int32),
                np.asarray(mstars, dtype=np.float64),
                np.asarray(binmax, dtype=np.float64),
                np.asarray(multi1, dtype=np.float64),
                np.asarray(tdead, dtype=np.float64),
                np.asarray(hecores, dtype=np.float64),
                np.asarray(mstars1_eff, dtype=np.float64),
                np.asarray(qispecial, dtype=np.float64),
                int(self.tables.ninputyield),
                self._W,
                self._massa,
                self._massac,
                self._MBa,
                self._WBa,
                self._WSr,
                self._WY,
                self._WLa,
                self._WZr,
                self._WRb,
                self._WEu,
                self._zbario,
                self._massaba,
                self._ba,
                self._sr,
                self._yt,
                self._eu,
                self._zr,
                self._la,
                self._rb,
                self._YLi,
                self._massaLi,
            )
        )


def build_cython_backend(tables: ModelTables, cfg: RunConfig | None = None):
    if _cyinterp is None:
        raise RuntimeError(
            "Cython backend unavailable: pyche._cyinterp extension is not built"
        ) from _CYTHON_IMPORT_ERROR
    if cfg is None:
        return CythonModelInterpolator(tables)
    return CythonModelInterpolator(
        tables,
        enable_lookup_cache=bool(cfg.interp_cache),
        cache_mass_points=int(cfg.interp_cache_mass_points),
        cache_zeta_points=int(cfg.interp_cache_zeta_points),
        cache_binmax_points=int(cfg.interp_cache_binmax_points),
        cache_zeta_max=float(cfg.interp_cache_zeta_max),
        enable_cache_guard=bool(cfg.interp_cache_guard),
        cache_guard_tol=float(cfg.interp_cache_guard_tol),
        cache_guard_stride=int(cfg.interp_cache_guard_stride),
        cache_guard_samples=int(cfg.interp_cache_guard_samples),
    )
