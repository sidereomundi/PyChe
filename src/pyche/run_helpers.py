"""Helpers for running and summarizing GCE outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_fis(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, skiprows=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def summarize_fis(fis: np.ndarray) -> dict[str, float]:
    return {
        "final_gas": float(fis[-1, 2]),
        "final_stars": float(fis[-1, 3]),
        "final_remn": float(fis[-1, 4]),
        "final_zeta": float(fis[-1, 6]),
        "max_sfr": float(fis[:, 7].max()),
    }

