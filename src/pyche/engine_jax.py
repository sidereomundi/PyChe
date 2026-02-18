"""JAX frontend for GCE engine execution.

This module isolates the JAX execution path so we can progressively move
runtime loop sections into JAX-native kernels without affecting the numpy/cython
paths.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import RunConfig
from .engine_core import run_mingce_loop
from .state import SimulationState

try:
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MPI = None


def run_mingce_jax(
    model: Any,
    cfg: RunConfig,
    runtime: SimulationState,
    *,
    mstars: np.ndarray,
    binmax: np.ndarray,
    multi1: np.ndarray,
    tdead: np.ndarray,
    norm: float,
    ss2: int,
    elem: int = 33,
) -> dict[str, np.ndarray] | None:
    """Run engine loop through the JAX backend-specific frontend.

    The current implementation reuses ``run_mingce_loop`` while reserving this
    entrypoint for incremental replacement with JAX-native timestep kernels.
    """
    comm, rank, size = model._mpi_ctx()
    mpi_active = bool(cfg.use_mpi and MPI is not None and comm is not None and size > 1)
    return run_mingce_loop(
        model,
        cfg,
        runtime,
        mstars=mstars,
        binmax=binmax,
        multi1=multi1,
        tdead=tdead,
        norm=norm,
        ss2=ss2,
        elem=elem,
        comm=comm,
        rank=rank,
        size=size,
        mpi_active=mpi_active,
        mpi_module=MPI,
    )
