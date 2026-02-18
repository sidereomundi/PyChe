"""Backend factory for runtime interpolation kernels."""

from __future__ import annotations

from .cython_backend import build_cython_backend
from .jax_backend import build_jax_backend
from .numba_backend import build_numba_backend
from .numpy_backend import build_numpy_backend
from ..config import RunConfig
from ..model_tables import ModelTables


def build_backend(name: str, tables: ModelTables, cfg: RunConfig | None = None):
    if name == "numpy":
        return build_numpy_backend(tables)
    if name == "cython":
        return build_cython_backend(tables, cfg=cfg)
    if name == "numba":
        return build_numba_backend(tables)
    if name == "jax":
        return build_jax_backend(tables, cfg=cfg)
    if name == "auto":
        try:
            return build_cython_backend(tables, cfg=cfg)
        except Exception:
            pass
        try:
            return build_numba_backend(tables)
        except Exception:
            pass
        try:
            return build_jax_backend(tables, cfg=cfg)
        except Exception:
            return build_numpy_backend(tables)
    raise ValueError(f"Unknown backend: {name}")
