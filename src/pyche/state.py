"""Runtime simulation state for GCE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationState:
    """Mutable arrays evolved during a single GCE run."""

    allv: np.ndarray
    gas: np.ndarray
    stars: np.ndarray
    remn: np.ndarray
    hot: np.ndarray
    wind: np.ndarray
    oldstars: np.ndarray
    zeta: np.ndarray
    snianum: np.ndarray
    spalla: np.ndarray
    sfr_hist: np.ndarray
    qqn: np.ndarray
    ini: np.ndarray
    winds: np.ndarray

    @classmethod
    def create(cls, nmax: int = 15000, elem: int = 33) -> "SimulationState":
        allv = np.zeros(nmax + 2, dtype=float)
        gas = np.zeros(nmax + 2, dtype=float)
        stars = np.zeros(nmax + 2, dtype=float)
        remn = np.zeros(nmax + 2, dtype=float)
        hot = np.zeros(nmax + 2, dtype=float)
        wind = np.zeros(nmax + 2, dtype=float)
        oldstars = np.zeros(nmax + 2, dtype=float)
        zeta = np.zeros(nmax + 2, dtype=float)
        snianum = np.zeros(nmax + 2, dtype=float)
        spalla = np.zeros(nmax + 2, dtype=float)
        sfr_hist = np.zeros(nmax + 2, dtype=float)
        qqn = np.full((elem + 1, nmax + 2), 1.0e-20, dtype=float)
        ini = np.zeros(elem + 1, dtype=float)
        ini[31] = 1.0e-9
        winds = np.ones(32, dtype=float)
        winds[9] = 1.0
        return cls(
            allv=allv,
            gas=gas,
            stars=stars,
            remn=remn,
            hot=hot,
            wind=wind,
            oldstars=oldstars,
            zeta=zeta,
            snianum=snianum,
            spalla=spalla,
            sfr_hist=sfr_hist,
            qqn=qqn,
            ini=ini,
            winds=winds,
        )

