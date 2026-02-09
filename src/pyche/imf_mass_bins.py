"""IMF setup and mass-bin construction extracted from MinGCE main module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_routines import DATA_DIR
from .tau import tau


@dataclass(frozen=True)
class IMFParams:
    UM1: float
    UM2: float
    A: float
    B: float
    M1: float


def _imf_scalo_params() -> IMFParams:
    um1 = -1.35
    um2 = -1.7
    m1 = 2.0
    ms = 100.0
    m3 = 1.0
    zita = 0.3
    azita = ms ** (1.0 + um2)
    bzita = m1 ** (1.0 + um2)
    czita = m1 ** (1.0 + um1)
    dzita = m3 ** (1.0 + um1)
    b = zita / (m1 ** (um2 - um1) * (czita - dzita) / (1.0 + um1) + (azita - bzita) / (1.0 + um2))
    a = b * m1 ** (um2 - um1)
    return IMFParams(UM1=um1, UM2=um2, A=a, B=b, M1=m1)


def _imf_salpeter_params() -> IMFParams:
    um1 = -1.35
    mi = 0.1
    ms = 80.0
    azita = ms ** (1.0 + um1)
    bzita = mi ** (1.0 + um1)
    a = (1.0 + um1) / (azita - bzita)
    return IMFParams(UM1=um1, UM2=0.0, A=a, B=0.0, M1=0.0)


def _multi_scalo(params: IMFParams, mmax: float, mmin: float) -> float:
    if mmax <= params.M1:
        return ((mmin**params.UM1 - mmax**params.UM1) / params.UM1) * params.A
    return ((mmin**params.UM2 - mmax**params.UM2) / params.UM2) * params.B


def _multi_salpeter(params: IMFParams, mmax: float, mmin: float) -> float:
    return ((mmin**params.UM1 - mmax**params.UM1) / params.UM1) * params.A


def build_mass_bins(imf: int, tautype: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Build mass bins and lifetimes exactly as in translated MinGCE."""
    amu_vals = np.loadtxt(DATA_DIR / "DATI" / "amu.dat").flatten()
    amu = np.zeros(116, dtype=float)
    amu[1:116] = amu_vals[:115]

    mstars = np.zeros(2000, dtype=float)
    mstars1 = np.zeros(2000, dtype=float)
    multi1 = np.zeros(2000, dtype=float)
    binmax = np.zeros(2000, dtype=float)
    binmax1 = np.zeros(2000, dtype=float)
    multi2 = np.zeros(2000, dtype=float)

    if imf == 1:
        params = _imf_scalo_params()
        multi_fn = _multi_scalo
    elif imf == 3:
        params = _imf_salpeter_params()
        multi_fn = _multi_salpeter
    else:
        raise NotImplementedError("IMFKroupa setup is not present in src/main.f90")

    norm = 0.0
    ss = 0
    binmass = 115
    ss2 = binmass - 1

    for jj in range(1, binmass):
        mstars1[jj] = amu[binmass - jj]
        mstars1[jj + 1] = amu[binmass - jj + 1]
        multi1[jj] = multi_fn(params, mstars1[jj], mstars1[jj + 1])

        mstars[jj] = 0.5 * (mstars1[jj] + mstars1[jj + 1])
        binmax[jj] = 0.0

        if 3.0 <= mstars[jj] <= 16.0:
            mmu = mstars[jj]
            mumin = 0.8 / mmu
            mumin2 = 1.0 - 8.0 / mmu
            if mumin2 > mumin:
                mumin = mumin2

            mux = 0.5 - mumin
            xmu = np.zeros(12, dtype=float)
            xmu[1] = mumin
            xmu[2] = mumin + 0.01 * mux
            xmu[3] = mumin + 0.02 * mux
            xmu[4] = mumin + 0.05 * mux
            xmu[5] = mumin + 0.1 * mux
            xmu[6] = mumin + 0.2 * mux
            xmu[7] = mumin + 0.3 * mux
            xmu[8] = mumin + 0.4 * mux
            xmu[9] = mumin + 0.6 * mux
            xmu[10] = mumin + 0.8 * mux
            xmu[11] = 0.5

            for j3 in range(1, 11):
                ss += 1
                ss2 = ss + binmass - 1
                mu1 = xmu[j3]
                mu2 = xmu[j3 + 1]
                mstars[ss2] = mstars[jj] * (xmu[j3] + xmu[j3]) / 2.0
                multi1[ss2] = (8.0 * (mu2**3 - mu1**3)) * multi1[jj] * 0.09
                binmax[ss2] = mstars[jj] - mstars[ss2]
                norm += multi1[ss2] * mstars[jj]

            multi1[jj] = 0.95 * multi1[jj]

        if 2.0 <= mstars[jj] <= 8.0:
            ss += 1
            ss2 = ss + binmass - 1
            mstars[ss2] = mstars[jj]
            binmax[ss2] = -1.0
            multi1[ss2] = 0.015 * multi1[jj]

        norm += multi1[jj] * mstars[jj]

    tdead_raw = np.zeros(ss2 + 1, dtype=float)
    for j in range(1, ss2 + 1):
        tdead_raw[j] = tau(max(mstars[j], 1.0e-8), tautype, binmax[j])

    order = np.argsort(tdead_raw[1 : ss2 + 1])
    for idx, o in enumerate(order, start=1):
        src = o + 1
        mstars1[idx] = mstars[src]
        binmax1[idx] = binmax[src]
        multi2[idx] = multi1[src]

    for jj in range(1, ss2 + 1):
        mstars[jj] = mstars1[jj]
        binmax[jj] = binmax1[jj]
        multi1[jj] = multi2[jj]

    tdead = np.zeros(2001, dtype=float)
    tdead[ss2 + 1] = 1.0e30
    for jj in range(1, ss2 + 1):
        tdead[jj] = tau(max(mstars[jj], 1.0e-8), tautype, binmax[jj])

    return mstars, binmax, multi1, tdead, norm, ss2
