"""Immutable table container for GCE inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelTables:
    """Loaded model tables in Fortran-compatible (1-based) array layout."""

    ninputyield: int
    Q: np.ndarray
    W: np.ndarray
    WH: np.ndarray
    massa: np.ndarray
    massac: np.ndarray
    massac2: np.ndarray
    massas: np.ndarray
    MBa: np.ndarray
    WBa: np.ndarray
    MSr: np.ndarray
    WSr: np.ndarray
    MY: np.ndarray
    WY: np.ndarray
    MLa: np.ndarray
    WLa: np.ndarray
    MRb: np.ndarray
    WRb: np.ndarray
    MZr: np.ndarray
    WZr: np.ndarray
    MEu: np.ndarray
    WEu: np.ndarray
    zbario: np.ndarray
    massaba: np.ndarray
    ba: np.ndarray
    sr: np.ndarray
    yt: np.ndarray
    eu: np.ndarray
    zr: np.ndarray
    la: np.ndarray
    rb: np.ndarray
    YLi: np.ndarray
    massaLi: np.ndarray

    @classmethod
    def from_state(cls, state: object) -> "ModelTables":
        return cls(
            ninputyield=int(state.ninputyield),
            Q=np.array(state.Q, copy=True),
            W=np.array(state.W, copy=True),
            WH=np.array(state.WH, copy=True),
            massa=np.array(state.massa, copy=True),
            massac=np.array(state.massac, copy=True),
            massac2=np.array(state.massac2, copy=True),
            massas=np.array(state.massas, copy=True),
            MBa=np.array(state.MBa, copy=True),
            WBa=np.array(state.WBa, copy=True),
            MSr=np.array(state.MSr, copy=True),
            WSr=np.array(state.WSr, copy=True),
            MY=np.array(state.MY, copy=True),
            WY=np.array(state.WY, copy=True),
            MLa=np.array(state.MLa, copy=True),
            WLa=np.array(state.WLa, copy=True),
            MRb=np.array(state.MRb, copy=True),
            WRb=np.array(state.WRb, copy=True),
            MZr=np.array(state.MZr, copy=True),
            WZr=np.array(state.WZr, copy=True),
            MEu=np.array(state.MEu, copy=True),
            WEu=np.array(state.WEu, copy=True),
            zbario=np.array(state.zbario, copy=True),
            massaba=np.array(state.massaba, copy=True),
            ba=np.array(state.ba, copy=True),
            sr=np.array(state.sr, copy=True),
            yt=np.array(state.yt, copy=True),
            eu=np.array(state.eu, copy=True),
            zr=np.array(state.zr, copy=True),
            la=np.array(state.la, copy=True),
            rb=np.array(state.rb, copy=True),
            YLi=np.array(state.YLi, copy=True),
            massaLi=np.array(state.massaLi, copy=True),
        )

