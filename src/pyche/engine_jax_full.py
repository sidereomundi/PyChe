"""Accurate JAX-full frontend for GCE execution.

This backend currently prioritizes physical accuracy/parity by reusing the
validated shared core loop while forcing JAX interpolation backend selection in
``main.GCE``.
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


def run_mingce_jax_full(
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
    """Run the full GCE evolution through the validated core loop.

    Notes:
    - This path expects ``model.interpolator`` to be a JAX interpolator built
      by ``main.GCE``.
    - Using the shared core loop keeps abundance tracks (including O/Fe and
      Mg/Fe) consistent with the reference implementation.
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
