"""Readers for legacy/dataframe MinGCE outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .output_io import FIS_COLUMNS, MOD_COLUMNS


def _read_legacy(out_dir: Path):
    mod = np.loadtxt(out_dir / "modencesmin.dat", skiprows=1)
    fis = np.loadtxt(out_dir / "fis.encesmin.dat", skiprows=1)
    if mod.ndim == 1:
        mod = mod[None, :]
    if fis.ndim == 1:
        fis = fis[None, :]
    return {"mod": mod, "fis": fis, "format": "legacy"}


def _read_dataframe(out_dir: Path, binary_format: str = "pickle"):
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Reading dataframe outputs requires pandas installed") from exc

    if binary_format == "pickle":
        mod = pd.read_pickle(out_dir / "modencesmin_df.pkl")
        fis = pd.read_pickle(out_dir / "fis.encesmin_df.pkl")
    else:
        mod = pd.read_parquet(out_dir / "modencesmin_df.parquet")
        fis = pd.read_parquet(out_dir / "fis.encesmin_df.parquet")
    return {"mod": mod, "fis": fis, "format": "dataframe"}


def read_outputs(output_dir: str | Path, *, prefer: str = "auto", binary_format: str = "pickle"):
    out_dir = Path(output_dir)
    if prefer == "legacy":
        return _read_legacy(out_dir)
    if prefer == "dataframe":
        return _read_dataframe(out_dir, binary_format=binary_format)

    # auto
    if (out_dir / "modencesmin_df.pkl").exists() or (out_dir / "modencesmin_df.parquet").exists():
        fmt = "parquet" if (out_dir / "modencesmin_df.parquet").exists() else "pickle"
        return _read_dataframe(out_dir, binary_format=fmt)
    return _read_legacy(out_dir)

