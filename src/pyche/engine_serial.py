"""Serial frontend for GCE engine execution."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import RunConfig
from .engine_core import run_mingce_loop
from .state import SimulationState


def run_mingce_serial(
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
        comm=None,
        rank=0,
        size=1,
        mpi_active=False,
        mpi_module=None,
    )
