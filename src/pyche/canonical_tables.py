"""Canonical table loader for the upcoming solver rewrite.

This module exposes a normalized, zero-based view of all yield tables while
reusing the validated Fortran-compatible loading path from ``io_routines``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_routines import FortranState, IORoutines


@dataclass(frozen=True)
class CanonicalTables:
    ninputyield: int
    mass_grid: np.ndarray
    yield_grid: np.ndarray
    cris_mass_grid: np.ndarray
    cris_yield_grid: np.ndarray
    barium_mass_grid: np.ndarray
    barium_z_grid: np.ndarray
    barium_species: dict[str, np.ndarray]
    massive_sprocess_mass_grid: np.ndarray
    massive_sprocess_yields: dict[str, np.ndarray]
    lithium_mass_grid: np.ndarray
    lithium_yields: np.ndarray


def load_canonical_tables(lowmassive: int = 1, mm: int = 0) -> CanonicalTables:
    """Load all model tables in a normalized zero-based representation."""
    io = IORoutines()
    state = FortranState()
    io.load_main_tables(state, lowmassive=lowmassive, mm=mm)

    n = state.ninputyield

    return CanonicalTables(
        ninputyield=n,
        mass_grid=state.massa[1 : n + 1].copy(),
        yield_grid=state.W[1:24, 1 : n + 1, 1:6].copy(),  # (23, nmass, 5z)
        cris_mass_grid=state.massac[1:14].copy(),
        cris_yield_grid=state.W[1:15, 1:14, 6:11].copy(),  # (14, 13mass, 5z)
        barium_mass_grid=state.massaba[1:6].copy(),
        barium_z_grid=state.zbario[1:10].copy(),
        barium_species={
            "Ba": state.ba[1:6, 1:10].copy(),
            "Sr": state.sr[1:6, 1:10].copy(),
            "Y": state.yt[1:6, 1:10].copy(),
            "Eu": state.eu[1:6, 1:10].copy(),
            "Zr": state.zr[1:6, 1:10].copy(),
            "La": state.la[1:6, 1:10].copy(),
            "Rb": state.rb[1:6, 1:10].copy(),
        },
        massive_sprocess_mass_grid=state.MBa[1:5].copy(),
        massive_sprocess_yields={
            "Ba": state.WBa[1:5, 1:4].copy(),
            "Sr": state.WSr[1:5, 1:4].copy(),
            "Y": state.WY[1:5, 1:4].copy(),
            "Eu": state.WEu[1:5, 1:4].copy(),
            "Zr": state.WZr[1:5, 1:4].copy(),
            "La": state.WLa[1:5, 1:4].copy(),
            "Rb": state.WRb[1:5, 1:4].copy(),
        },
        lithium_mass_grid=state.massaLi[1:16].copy(),
        lithium_yields=state.YLi[1:16, 1:5].copy(),
    )
