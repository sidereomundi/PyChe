"""Main MinGCE routine translated from ``src/main.f90``."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import numpy as np

from .backends.factory import build_backend
from .config import RunConfig
from .engine import run_mingce
from .imf_mass_bins import build_mass_bins
from .io_routines import FortranState, IORoutines
from .model_tables import ModelTables
from .output_io import FIS_COLUMNS, MOD_COLUMNS
from .state import SimulationState

try:
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MPI = None


@dataclass
class MinGCEResult:
    mod: np.ndarray
    fis: np.ndarray
    mod_columns: tuple[str, ...] = tuple(MOD_COLUMNS)
    fis_columns: tuple[str, ...] = tuple(FIS_COLUMNS)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"mod": self.mod, "fis": self.fis}


@dataclass
class GCEModel:
    io: IORoutines = field(default_factory=IORoutines)
    state: FortranState = field(default_factory=FortranState)
    tables: ModelTables | None = None
    interpolator: object = field(init=False)

    def __post_init__(self) -> None:
        self.interpolator = None

    def _initialize_from_fortran_tables(self, lowmassive: int = 1, mm: int = 0) -> None:
        self.tables = self.io.load_model_tables(lowmassive=lowmassive, mm=mm)

    def _mpi_ctx(self) -> tuple[object | None, int, int]:
        if MPI is None:
            return None, 0, 1
        comm = MPI.COMM_WORLD
        return comm, int(comm.Get_rank()), int(comm.Get_size())

    def MinGCE(
        self,
        endoftime: int,
        sigmat: float,
        sigmah: float,
        psfr: float,
        pwind: float,
        delay: int,
        time_wind: int,
        use_mpi: bool = True,
        mpi_nonblocking_reduce: bool = False,
        show_progress: bool = True,
        output_dir: str | None = None,
        output_mode: str = "legacy",
        write_output: bool = True,
        df_binary_format: str = "pickle",
        df_write_csv: bool = False,
        backend: str = "auto",
        adaptive_timestep: bool = True,
        dt_min: int = 1,
        dt_max: int = 5,
        dt_rel_tol: float = 0.1,
        dt_smooth_alpha: float = 0.3,
        dt_growth_factor: float = 1.25,
        dt_shrink_factor: float = 0.5,
        spalla_stride: int = 4,
        spalla_inactive_threshold: float = 1.0e-12,
        spalla_lut: bool = True,
        spalla_lut_q_points: int = 128,
        spalla_lut_d_points: int = 128,
        spalla_lut_logq_min: float = -30.0,
        spalla_lut_logq_max: float = 2.0,
        spalla_lut_logd_min: float = -30.0,
        spalla_lut_logd_max: float = 2.0,
        interp_cache: bool = True,
        interp_cache_mass_points: int = 96,
        interp_cache_zeta_points: int = 64,
        interp_cache_binmax_points: int = 64,
        interp_cache_zeta_max: float = 0.05,
        interp_cache_guard: bool = True,
        interp_cache_guard_tol: float = 0.05,
        interp_cache_guard_stride: int = 16,
        profile_timing: bool = True,
        return_results: bool = False,
    ) -> MinGCEResult | None:
        cfg = RunConfig(
            endoftime=endoftime,
            sigmat=sigmat,
            sigmah=sigmah,
            psfr=psfr,
            pwind=pwind,
            delay=delay,
            time_wind=time_wind,
            use_mpi=use_mpi,
            mpi_nonblocking_reduce=mpi_nonblocking_reduce,
            show_progress=show_progress,
            output_dir=output_dir,
            output_mode=output_mode,
            write_output=write_output,
            df_binary_format=df_binary_format,
            df_write_csv=df_write_csv,
            backend=backend,
            adaptive_timestep=adaptive_timestep,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_rel_tol=dt_rel_tol,
            dt_smooth_alpha=dt_smooth_alpha,
            dt_growth_factor=dt_growth_factor,
            dt_shrink_factor=dt_shrink_factor,
            spalla_stride=spalla_stride,
            spalla_inactive_threshold=spalla_inactive_threshold,
            spalla_lut=spalla_lut,
            spalla_lut_q_points=spalla_lut_q_points,
            spalla_lut_d_points=spalla_lut_d_points,
            spalla_lut_logq_min=spalla_lut_logq_min,
            spalla_lut_logq_max=spalla_lut_logq_max,
            spalla_lut_logd_min=spalla_lut_logd_min,
            spalla_lut_logd_max=spalla_lut_logd_max,
            interp_cache=interp_cache,
            interp_cache_mass_points=interp_cache_mass_points,
            interp_cache_zeta_points=interp_cache_zeta_points,
            interp_cache_binmax_points=interp_cache_binmax_points,
            interp_cache_zeta_max=interp_cache_zeta_max,
            interp_cache_guard=interp_cache_guard,
            interp_cache_guard_tol=interp_cache_guard_tol,
            interp_cache_guard_stride=interp_cache_guard_stride,
            profile_timing=profile_timing,
        )

        nmax = 15000
        elem = 33

        self._initialize_from_fortran_tables(lowmassive=1, mm=0)
        self.interpolator = build_backend(cfg.backend, self.tables, cfg=cfg)

        imf = 1
        tautype = 1
        mstars, binmax, multi1, tdead, norm, ss2 = build_mass_bins(imf, tautype)

        runtime = SimulationState.create(nmax=nmax, elem=elem)
        t0 = time.perf_counter()
        payload = run_mingce(
            self,
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
        elapsed = time.perf_counter() - t0

        comm, rank, size = self._mpi_ctx()
        if cfg.use_mpi and comm is not None and size > 1:
            max_elapsed = comm.reduce(elapsed, op=MPI.MAX, root=0) if MPI is not None else elapsed
            if rank == 0:
                print(f"total runtime (wall, max-rank): {float(max_elapsed):.3f} s")
        else:
            print(f"total runtime (wall): {elapsed:.3f} s")
        if return_results:
            if payload is None:
                return None
            return MinGCEResult(mod=payload["mod_rows"], fis=payload["fis_rows"])
        return None
