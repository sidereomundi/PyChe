"""Fortran-style IO translation for GCE tables.

This module translates the file-loading routines from ``src/io.f90`` and the
main table-loading block in ``src/main.f90``.  Array shapes keep a leading
unused slot so 1-based Fortran indices map directly to Python indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
import hashlib
import json
import os
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
BASE_DIR = PACKAGE_DIR
TABLE_CACHE_VERSION = 1


@dataclass
class FortranState:
    """Container mirroring the Fortran COMMON blocks used by GCE."""

    ninputyield: int = 0
    Q: np.ndarray = field(default_factory=lambda: np.zeros(34, dtype=float))
    W: np.ndarray = field(default_factory=lambda: np.zeros((24, 36, 16), dtype=float))
    WH: np.ndarray = field(default_factory=lambda: np.zeros((24, 36, 16), dtype=float))
    massa: np.ndarray = field(default_factory=lambda: np.zeros(36, dtype=float))
    massac: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=float))
    massac2: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))
    massas: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=float))

    MBa: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))
    WBa: np.ndarray = field(default_factory=lambda: np.zeros((5, 4), dtype=float))
    MSr: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))
    WSr: np.ndarray = field(default_factory=lambda: np.zeros((5, 4), dtype=float))
    MY: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))
    WY: np.ndarray = field(default_factory=lambda: np.zeros((5, 4), dtype=float))
    MLa: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))
    WLa: np.ndarray = field(default_factory=lambda: np.zeros((5, 4), dtype=float))
    MRb: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))
    WRb: np.ndarray = field(default_factory=lambda: np.zeros((5, 4), dtype=float))
    MZr: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))
    WZr: np.ndarray = field(default_factory=lambda: np.zeros((5, 4), dtype=float))
    MEu: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))
    WEu: np.ndarray = field(default_factory=lambda: np.zeros((5, 4), dtype=float))

    zbario: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=float))
    massaba: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    ba: np.ndarray = field(default_factory=lambda: np.zeros((6, 10), dtype=float))
    sr: np.ndarray = field(default_factory=lambda: np.zeros((6, 10), dtype=float))
    yt: np.ndarray = field(default_factory=lambda: np.zeros((6, 10), dtype=float))
    eu: np.ndarray = field(default_factory=lambda: np.zeros((6, 10), dtype=float))
    zr: np.ndarray = field(default_factory=lambda: np.zeros((6, 10), dtype=float))
    la: np.ndarray = field(default_factory=lambda: np.zeros((6, 10), dtype=float))
    rb: np.ndarray = field(default_factory=lambda: np.zeros((6, 10), dtype=float))

    YLi: np.ndarray = field(default_factory=lambda: np.zeros((16, 5), dtype=float))
    massaLi: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=float))


@dataclass
class IORoutines:
    base_dati: Path = field(default_factory=lambda: DATA_DIR / "DATI")
    base_yieldsba: Path = field(default_factory=lambda: DATA_DIR / "YIELDSBA")
    enable_table_cache: bool = True
    table_cache_dir: Path | None = None

    def _resolve_table_cache_dir(self) -> Path:
        if self.table_cache_dir is not None:
            return Path(self.table_cache_dir)
        env_dir = os.getenv("PYCHE_TABLE_CACHE_DIR")
        if env_dir:
            return Path(env_dir).expanduser()
        return Path.home() / ".cache" / "pyche"

    def _table_source_files(self, lowmassive: int, mm: int) -> list[Path]:
        if lowmassive == 1:
            inputyield = ["WW95aMD.csv", "WW95bMD.csv", "WW95cMD.csv", "WW95dMD.csv", "WW95e.csv"]
            if mm == 1:
                inputyield[4] = "WW95eMM.csv"
            elif mm == 2:
                inputyield[4] = "WW95eMD.csv"
        else:
            inputyield = [
                "WW95zeroMD.K0001.csv",
                "WW95-4MD.K004.csv",
                "WW95-2MD.K008.csv",
                "WW95-1MD.K02.csv",
                "WW9502.K02.csv",
            ]
            if mm == 1:
                inputyield[4] = "WW9502MM.K02.csv"
            elif mm == 2:
                inputyield[4] = "WW9502MD.K02.csv"

        dati_files = inputyield + [
            "Kobayashi-Iron.dat",
            "Kobayashi-IronHyper.dat",
            "Cris0.dat",
            "Cris8.dat",
            "Cris5.dat",
            "Cris004.dat",
            "Cris02.dat",
            "Bariumnew.dat",
            "Strontiumnew.dat",
            "Yttriumnew.dat",
            "Lantanumnew.dat",
            "Zirconiumnew.dat",
            "Rubidiumnew.dat",
            "Europiumnew.dat",
        ]
        yieldsba_files = [
            "CristalloBa2.dat",
            "CristalloSr.dat",
            "CristalloY.dat",
            "CristalloEu.dat",
            "CristalloZr.dat",
            "CristalloLa.dat",
            "CristalloRb.dat",
            "KarakasLi.dat",
        ]
        return [self.base_dati / name for name in dati_files] + [self.base_yieldsba / name for name in yieldsba_files]

    def _table_cache_token(self, lowmassive: int, mm: int) -> str:
        entries: list[tuple[str, int, int]] = []
        for path in self._table_source_files(lowmassive=lowmassive, mm=mm):
            st = path.stat()
            entries.append((path.name, int(st.st_size), int(st.st_mtime_ns)))
        payload = {
            "version": TABLE_CACHE_VERSION,
            "lowmassive": int(lowmassive),
            "mm": int(mm),
            "sources": entries,
        }
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _table_cache_path(self, lowmassive: int, mm: int) -> Path:
        token = self._table_cache_token(lowmassive=lowmassive, mm=mm)
        cache_dir = self._resolve_table_cache_dir()
        return cache_dir / f"model_tables_v{TABLE_CACHE_VERSION}_lm{int(lowmassive)}_mm{int(mm)}_{token}.npz"

    def _load_table_cache(self, lowmassive: int, mm: int):
        from .model_tables import ModelTables

        path = self._table_cache_path(lowmassive=lowmassive, mm=mm)
        if not path.exists():
            return None

        try:
            with np.load(path, allow_pickle=False) as data:
                kwargs = {}
                for f in dataclass_fields(ModelTables):
                    if f.name not in data:
                        return None
                    if f.name == "ninputyield":
                        kwargs[f.name] = int(data[f.name])
                    else:
                        kwargs[f.name] = np.asarray(data[f.name], dtype=float)
            return ModelTables(**kwargs)
        except Exception:
            return None

    def _save_table_cache(self, lowmassive: int, mm: int, tables: object) -> None:
        path = self._table_cache_path(lowmassive=lowmassive, mm=mm)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for f in dataclass_fields(type(tables)):
            value = getattr(tables, f.name)
            if f.name == "ninputyield":
                payload[f.name] = np.array(int(value), dtype=np.int64)
            else:
                payload[f.name] = np.asarray(value)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **payload)
        tmp.replace(path)

    def _read_numeric_table(self, path: Path, skiprows: int = 0) -> np.ndarray:
        return np.loadtxt(path, skiprows=skiprows)

    def _load_yield_grid_file(self, state: FortranState, z: int, filename: str) -> None:
        arr = self._read_numeric_table(self.base_dati / filename, skiprows=1)
        n = state.ninputyield
        state.massa[1 : n + 1] = arr[:n, 0]
        state.W[1:24, 1 : n + 1, z] = arr[:n, 1:24].T

    def _load_cris_file(self, state: FortranState, z: int, filename: str) -> None:
        arr = self._read_numeric_table(self.base_dati / filename, skiprows=2)
        state.massac[1:14] = arr[:13, 0]
        state.W[1:15, 1:14, z] = arr[:13, 1:15].T

    def leggi(self, state: FortranState) -> None:
        arr = self._read_numeric_table(self.base_yieldsba / "CristalloBa2.dat", skiprows=1)
        idx = 0
        for j in range(1, 6):
            for i in range(1, 10):
                row = arr[idx]
                idx += 1
                state.massaba[j] = row[0]
                state.zbario[i] = row[1]
                state.ba[j, i] = float(np.sum(row[2:9]))

    def leggiSr(self, state: FortranState) -> None:
        arr = self._read_numeric_table(self.base_yieldsba / "CristalloSr.dat", skiprows=1)
        idx = 0
        for j in range(1, 6):
            for i in range(1, 10):
                row = arr[idx]
                idx += 1
                state.massaba[j] = row[0]
                state.zbario[i] = row[1]
                state.sr[j, i] = float(np.sum(row[2:5]))

    def leggiY(self, state: FortranState) -> None:
        arr = self._read_numeric_table(self.base_yieldsba / "CristalloY.dat", skiprows=1)
        idx = 0
        for j in range(1, 6):
            for i in range(1, 10):
                row = arr[idx]
                idx += 1
                state.massaba[j] = row[0]
                state.zbario[i] = row[1]
                state.yt[j, i] = row[2]

    def leggiEu(self, state: FortranState) -> None:
        arr = self._read_numeric_table(self.base_yieldsba / "CristalloEu.dat", skiprows=1)
        idx = 0
        for j in range(1, 6):
            for i in range(1, 10):
                row = arr[idx]
                idx += 1
                state.massaba[j] = row[0]
                state.zbario[i] = row[1]
                state.eu[j, i] = float(np.sum(row[2:9]))

    def leggiZr(self, state: FortranState) -> None:
        arr = self._read_numeric_table(self.base_yieldsba / "CristalloZr.dat", skiprows=1)
        idx = 0
        for j in range(1, 6):
            for i in range(1, 10):
                row = arr[idx]
                idx += 1
                state.massaba[j] = row[0]
                state.zbario[i] = row[1]
                state.zr[j, i] = float(np.sum(row[2:10]))

    def leggiLa(self, state: FortranState) -> None:
        arr = self._read_numeric_table(self.base_yieldsba / "CristalloLa.dat", skiprows=1)
        idx = 0
        for j in range(1, 6):
            for i in range(1, 10):
                row = arr[idx]
                idx += 1
                state.massaba[j] = row[0]
                state.zbario[i] = row[1]
                state.la[j, i] = row[2]

    def leggiRb(self, state: FortranState) -> None:
        arr = self._read_numeric_table(self.base_yieldsba / "CristalloRb.dat", skiprows=1)
        idx = 0
        for j in range(1, 6):
            for i in range(1, 10):
                row = arr[idx]
                idx += 1
                state.massaba[j] = row[0]
                state.zbario[i] = row[1]
                state.rb[j, i] = float(np.sum(row[2:6]))

    def leggiLi(self, state: FortranState) -> None:
        arr = self._read_numeric_table(self.base_yieldsba / "KarakasLi.dat", skiprows=1)
        for i in range(1, 16):
            row = arr[i - 1]
            state.massaLi[i] = row[0]
            state.YLi[i, 1:5] = row[1:5]

    def load_main_tables(self, state: FortranState, lowmassive: int = 1, mm: int = 0) -> None:
        if lowmassive == 1:
            state.ninputyield = 32
            inputyield = [
                "WW95aMD.csv",
                "WW95bMD.csv",
                "WW95cMD.csv",
                "WW95dMD.csv",
                "WW95e.csv",
            ]
            if mm == 1:
                inputyield[4] = "WW95eMM.csv"
            if mm == 2:
                inputyield[4] = "WW95eMD.csv"
        else:
            state.ninputyield = 35
            inputyield = [
                "WW95zeroMD.K0001.csv",
                "WW95-4MD.K004.csv",
                "WW95-2MD.K008.csv",
                "WW95-1MD.K02.csv",
                "WW9502.K02.csv",
            ]
            if mm == 1:
                inputyield[4] = "WW9502MM.K02.csv"
            if mm == 2:
                inputyield[4] = "WW9502MD.K02.csv"

        for z, filename in enumerate(inputyield, start=1):
            self._load_yield_grid_file(state, z, filename)

        iron = self._read_numeric_table(self.base_dati / "Kobayashi-Iron.dat", skiprows=1)
        for n in range(15, 33):
            row = iron[n - 15]
            state.massa[n] = row[0]
            state.W[9, n, 5] = row[1]
            state.W[9, n, 4] = row[2]
            state.W[9, n, 3] = row[3]
            state.W[9, n, 2] = row[4]
            state.W[9, n, 1] = row[5]

        iron_h = self._read_numeric_table(self.base_dati / "Kobayashi-IronHyper.dat", skiprows=1)
        for n in range(15, 33):
            row = iron_h[n - 15]
            state.massa[n] = row[0]
            state.WH[9, n, 5] = row[1]
            state.WH[9, n, 4] = row[2]
            state.WH[9, n, 3] = row[3]
            state.WH[9, n, 2] = row[4]
            state.WH[9, n, 1] = row[5]
            for i in range(1, 6):
                state.W[9, n, i] = 0.5 * (state.WH[9, n, i] + state.W[9, n, i])

        self._load_cris_file(state, 6, "Cris0.dat")
        self._load_cris_file(state, 7, "Cris8.dat")
        self._load_cris_file(state, 8, "Cris5.dat")
        self._load_cris_file(state, 9, "Cris004.dat")
        self._load_cris_file(state, 10, "Cris02.dat")

        arr = self._read_numeric_table(self.base_dati / "Bariumnew.dat")
        for n in range(1, 5):
            state.MBa[n] = arr[n - 1, 0]
            state.WBa[n, 1:4] = arr[n - 1, 1:4]
            state.WBa[n, 3] *= 640.0

        arr = self._read_numeric_table(self.base_dati / "Strontiumnew.dat")
        for n in range(1, 5):
            state.MSr[n] = arr[n - 1, 0]
            state.WSr[n, 1:4] = arr[n - 1, 1:4]
            state.WSr[n, 3] *= 50.0

        arr = self._read_numeric_table(self.base_dati / "Yttriumnew.dat")
        for n in range(1, 5):
            state.MY[n] = arr[n - 1, 0]
            state.WY[n, 1:4] = arr[n - 1, 1:4]
            state.WY[n, 3] *= 150.0

        arr = self._read_numeric_table(self.base_dati / "Lantanumnew.dat")
        for n in range(1, 5):
            state.MLa[n] = arr[n - 1, 0]
            state.WLa[n, 1:4] = arr[n - 1, 1:4]
            state.WLa[n, 3] *= 211.0

        arr = self._read_numeric_table(self.base_dati / "Zirconiumnew.dat")
        for n in range(1, 5):
            state.MZr[n] = arr[n - 1, 0]
            state.WZr[n, 1:4] = arr[n - 1, 1:4]
            state.WZr[n, 3] *= 346.0

        arr = self._read_numeric_table(self.base_dati / "Rubidiumnew.dat")
        for n in range(1, 5):
            state.MRb[n] = arr[n - 1, 0]
            state.WRb[n, 1:4] = arr[n - 1, 1:4]
            state.WRb[n, 3] *= 10.0

        arr = self._read_numeric_table(self.base_dati / "Europiumnew.dat")
        for n in range(1, 5):
            state.MEu[n] = arr[n - 1, 0]
            state.WEu[n, 1:4] = arr[n - 1, 1:4]
            state.WEu[n, 3] *= 1.0e-20
            state.WEu[n, 2] = state.WEu[n, 3] * 1.0e-20
            state.WEu[n, 1] = state.WEu[n, 3] * 1.0e-20

        self.leggiLi(state)
        self.leggiLa(state)
        self.leggiRb(state)
        self.leggiZr(state)
        self.leggiEu(state)
        self.leggiY(state)
        self.leggiSr(state)
        self.leggi(state)

    def load_model_tables(self, lowmassive: int = 1, mm: int = 0):
        """Load tables and return immutable ``ModelTables`` representation."""
        from .model_tables import ModelTables

        if self.enable_table_cache:
            cached = self._load_table_cache(lowmassive=lowmassive, mm=mm)
            if cached is not None:
                return cached

        state = FortranState()
        self.load_main_tables(state, lowmassive=lowmassive, mm=mm)
        tables = ModelTables.from_state(state)
        if self.enable_table_cache:
            try:
                self._save_table_cache(lowmassive=lowmassive, mm=mm, tables=tables)
            except Exception:
                pass
        return tables
