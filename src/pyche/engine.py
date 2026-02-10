"""Engine dispatcher for GCE execution."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import RunConfig
from .engine_mpi import run_mingce_mpi
from .engine_serial import run_mingce_serial
from .state import SimulationState


def run_mingce(
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
    """Dispatch GCE execution to serial or MPI frontend."""
    if cfg.use_mpi:
        return run_mingce_mpi(
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
        )
    else:
        return run_mingce_serial(
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
        )
