"""Main GCE routine translated from ``src/main.f90``."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import pickle
import time
import subprocess
import sys
import tempfile
import numpy as np

from .backends.factory import build_backend
from .config import RunConfig
from .constants import NMAX_DEFAULT, NUM_ELEMENTS
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


def _build_runconfig_dict(
    *,
    endoftime_eff: int,
    sigmat: float, sigmah: float, psfr: float, pwind: float,
    delay: int, time_wind: int,
    input_time_arr: np.ndarray | None,
    infall_values_arr: np.ndarray | None,
    rhosfr_values_arr: np.ndarray | None,
    use_mpi: bool, mpi_nonblocking_reduce: bool,
    show_progress: bool, progress_style: str,
    output_dir: str | None, output_mode: str,
    write_output: bool,
    df_binary_format: str, df_write_csv: bool,
    backend: str,
    adaptive_timestep: bool,
    dt_min: int, dt_max: int, dt_rel_tol: float,
    dt_smooth_alpha: float, dt_growth_factor: float, dt_shrink_factor: float,
    dt_force_small_below_zeta: float, dt_force_small_value: int,
    spalla_stride: int, spalla_inactive_threshold: float,
    spalla_lut: bool,
    spalla_lut_q_points: int, spalla_lut_d_points: int,
    spalla_lut_logq_min: float, spalla_lut_logq_max: float,
    spalla_lut_logd_min: float, spalla_lut_logd_max: float,
    interp_cache: bool,
    interp_cache_mass_points: int, interp_cache_zeta_points: int,
    interp_cache_binmax_points: int, interp_cache_zeta_max: float,
    interp_cache_guard: bool, interp_cache_guard_tol: float,
    interp_cache_guard_stride: int, interp_cache_guard_samples: int,
    interp_cache_guard_force_below_zeta: float,
    interp_cache_guard_zeta_trigger: float,
    profile_timing: bool,
) -> dict[str, object]:
    """Build the parameter dict used for both MPI subprocess JSON and RunConfig."""
    return {
        "endoftime": endoftime_eff,
        "sigmat": sigmat, "sigmah": sigmah, "psfr": psfr, "pwind": pwind,
        "delay": delay, "time_wind": time_wind,
        "input_time": None if input_time_arr is None else [float(x) for x in input_time_arr],
        "infall_values": None if infall_values_arr is None else [float(x) for x in infall_values_arr],
        "rhosfr_values": None if rhosfr_values_arr is None else [float(x) for x in rhosfr_values_arr],
        "use_mpi": use_mpi, "mpi_nonblocking_reduce": mpi_nonblocking_reduce,
        "show_progress": show_progress, "progress_style": progress_style,
        "output_dir": output_dir, "output_mode": output_mode,
        "write_output": write_output,
        "df_binary_format": df_binary_format, "df_write_csv": df_write_csv,
        "backend": backend,
        "adaptive_timestep": adaptive_timestep,
        "dt_min": dt_min, "dt_max": dt_max, "dt_rel_tol": dt_rel_tol,
        "dt_smooth_alpha": dt_smooth_alpha,
        "dt_growth_factor": dt_growth_factor, "dt_shrink_factor": dt_shrink_factor,
        "dt_force_small_below_zeta": dt_force_small_below_zeta,
        "dt_force_small_value": dt_force_small_value,
        "spalla_stride": spalla_stride, "spalla_inactive_threshold": spalla_inactive_threshold,
        "spalla_lut": spalla_lut,
        "spalla_lut_q_points": spalla_lut_q_points, "spalla_lut_d_points": spalla_lut_d_points,
        "spalla_lut_logq_min": spalla_lut_logq_min, "spalla_lut_logq_max": spalla_lut_logq_max,
        "spalla_lut_logd_min": spalla_lut_logd_min, "spalla_lut_logd_max": spalla_lut_logd_max,
        "interp_cache": interp_cache,
        "interp_cache_mass_points": interp_cache_mass_points,
        "interp_cache_zeta_points": interp_cache_zeta_points,
        "interp_cache_binmax_points": interp_cache_binmax_points,
        "interp_cache_zeta_max": interp_cache_zeta_max,
        "interp_cache_guard": interp_cache_guard,
        "interp_cache_guard_tol": interp_cache_guard_tol,
        "interp_cache_guard_stride": interp_cache_guard_stride,
        "interp_cache_guard_samples": interp_cache_guard_samples,
        "interp_cache_guard_force_below_zeta": interp_cache_guard_force_below_zeta,
        "interp_cache_guard_zeta_trigger": interp_cache_guard_zeta_trigger,
        "profile_timing": profile_timing,
    }


@dataclass
class GCEResult:
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
    _interpolator_key: tuple[object, ...] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.interpolator = None
        self._interpolator_key = None

    def _initialize_from_fortran_tables(self, lowmassive: int = 1, mm: int = 0) -> None:
        if self.tables is None:
            self.tables = self.io.load_model_tables(lowmassive=lowmassive, mm=mm)

    @staticmethod
    def _backend_key(cfg: RunConfig) -> tuple[object, ...]:
        return (
            cfg.backend,
            cfg.interp_cache,
            cfg.interp_cache_mass_points,
            cfg.interp_cache_zeta_points,
            cfg.interp_cache_binmax_points,
            cfg.interp_cache_zeta_max,
            cfg.interp_cache_guard,
            cfg.interp_cache_guard_tol,
            cfg.interp_cache_guard_stride,
            cfg.interp_cache_guard_samples,
            cfg.interp_cache_guard_force_below_zeta,
            cfg.interp_cache_guard_zeta_trigger,
        )

    def _mpi_ctx(self) -> tuple[object | None, int, int]:
        if MPI is None:
            return None, 0, 1
        comm = MPI.COMM_WORLD
        return comm, int(comm.Get_rank()), int(comm.Get_size())

    def _run_mpi_subprocess(self, mpi_subprocess_ranks: int, kwargs: dict[str, object]) -> GCEResult:
        if mpi_subprocess_ranks < 2:
            raise ValueError("mpi_subprocess_ranks must be >= 2")
        show_progress = bool(kwargs.get("show_progress", False))
        progress_style = str(kwargs.get("progress_style", "single")).strip().lower()
        endoftime = int(kwargs.get("endoftime", 0))
        if show_progress:
            kwargs = dict(kwargs)
            kwargs["show_progress"] = False
        fd_payload, payload_path = tempfile.mkstemp(prefix="pyche_mpi_payload_", suffix=".json")
        try:
            with os.fdopen(fd_payload, "w", encoding="utf-8") as f_payload:
                json.dump(kwargs, f_payload)
        except Exception:
            try:
                os.remove(payload_path)
            except OSError:
                pass
            raise
        fd, result_path = tempfile.mkstemp(prefix="pyche_mpi_result_", suffix=".pkl")
        os.close(fd)
        fd_p, progress_path = tempfile.mkstemp(prefix="pyche_mpi_progress_", suffix=".txt")
        os.close(fd_p)
        cmd = [
            "mpiexec",
            "-n",
            str(int(mpi_subprocess_ranks)),
            sys.executable,
            "-u",
            "-m",
            "pyche._mpiexec_bridge",
            f"@{payload_path}",
        ]
        env = dict(os.environ)
        env["PYCHE_MPI_RESULT_PATH"] = result_path
        env["PYCHE_PROGRESS_PATH"] = progress_path
        if progress_style not in {"auto", "single", "line", "compact", "off"}:
            raise ValueError("progress_style must be one of: auto, single, line, compact, off")
        notebook_mode = "ipykernel" in sys.modules
        if progress_style == "auto":
            progress_style = "compact" if notebook_mode else "single"
        progress_use_carriage = progress_style == "single"
        display_handle = None
        display_fn = None
        display_handle_cls = None
        use_ipy_display = notebook_mode and progress_style == "single"
        try:
            if use_ipy_display:
                from IPython.display import DisplayHandle, display  # type: ignore

                display_fn = display
                display_handle_cls = DisplayHandle
        except Exception:
            use_ipy_display = False
        last_pct = -1
        last_compact_bucket = -1

        def _print_parent_progress(step: int) -> None:
            nonlocal last_pct, display_handle, last_compact_bucket
            if not show_progress or endoftime <= 0:
                return
            if progress_style == "off":
                return
            step = max(0, min(int(step), endoftime))
            pct = int((100.0 * step) / endoftime)
            if pct == last_pct:
                return
            if progress_style == "compact":
                bucket = pct // 5
                if bucket == last_compact_bucket and pct < 100:
                    return
                last_compact_bucket = bucket
            last_pct = pct
            bar_w = 30
            fill = int(bar_w * step / endoftime)
            bar = "#" * fill + "-" * (bar_w - fill)
            msg = f"Progress [{bar}] {100.0 * step / endoftime:6.2f}% ({step}/{endoftime})"
            if use_ipy_display:
                if display_handle is None:
                    if display_handle_cls is not None:
                        display_handle = display_handle_cls()
                        display_handle.display(msg)
                    elif display_fn is not None:
                        display_fn(msg)
                else:
                    display_handle.update(msg)
            elif progress_use_carriage:
                sys.stdout.write(f"\r{msg}")
                sys.stdout.flush()
            else:
                print(msg, flush=True)
        try:
            proc = subprocess.Popen(cmd, env=env)
            while True:
                rc = proc.poll()
                if rc is not None:
                    if rc != 0:
                        raise RuntimeError(f"mpiexec failed with code {rc}")
                    break
                try:
                    with open(progress_path, "r", encoding="utf-8") as f:
                        raw = f.read().strip()
                    if raw:
                        _print_parent_progress(int(raw))
                except (OSError, ValueError):
                    pass
                time.sleep(0.25)
            if show_progress and endoftime > 0:
                if last_pct < 100:
                    _print_parent_progress(endoftime)
                sys.stdout.write("\n")
                sys.stdout.flush()
            if not os.path.exists(result_path):
                raise RuntimeError("MPI subprocess did not produce a result payload")
            with open(result_path, "rb") as f:
                obj = pickle.load(f)
            if not isinstance(obj, GCEResult):
                raise RuntimeError("MPI subprocess returned invalid result payload")
            return obj
        finally:
            try:
                os.remove(result_path)
            except OSError:
                pass
            try:
                os.remove(progress_path)
            except OSError:
                pass
            try:
                os.remove(payload_path)
            except OSError:
                pass

    def GCE(
        self,
        endoftime: int,
        sigmat: float,
        sigmah: float,
        psfr: float,
        pwind: float,
        delay: int,
        time_wind: int,
        input_time: np.ndarray | list[float] | tuple[float, ...] | None = None,
        infall_time: np.ndarray | list[float] | tuple[float, ...] | None = None,
        infall_values: np.ndarray | list[float] | tuple[float, ...] | None = None,
        rhosfr_values: np.ndarray | list[float] | tuple[float, ...] | None = None,
        use_mpi: bool = True,
        mpi_nonblocking_reduce: bool = False,
        show_progress: bool = True,
        progress_style: str = "single",
        output_dir: str | None = None,
        output_mode: str = "dataframe",
        write_output: bool = True,
        df_binary_format: str = "pickle",
        df_write_csv: bool = False,
        backend: str = "auto",
        adaptive_timestep: bool = True,
        dt_min: int = 1,
        dt_max: int = 10,
        dt_rel_tol: float = 0.2,
        dt_smooth_alpha: float = 0.3,
        dt_growth_factor: float = 1.5,
        dt_shrink_factor: float = 0.5,
        dt_force_small_below_zeta: float = 1.0e-4,
        dt_force_small_value: int = 1,
        spalla_stride: int = 4,
        spalla_inactive_threshold: float = 1.0e-12,
        spalla_lut: bool = True,
        spalla_lut_q_points: int = 128,
        spalla_lut_d_points: int = 128,
        spalla_lut_logq_min: float = -30.0,
        spalla_lut_logq_max: float = 2.0,
        spalla_lut_logd_min: float = -30.0,
        spalla_lut_logd_max: float = 2.0,
        interp_cache: bool = False,
        interp_cache_mass_points: int = 96,
        interp_cache_zeta_points: int = 64,
        interp_cache_binmax_points: int = 64,
        interp_cache_zeta_max: float = 0.05,
        interp_cache_guard: bool = True,
        interp_cache_guard_tol: float = 0.05,
        interp_cache_guard_stride: int = 16,
        interp_cache_guard_samples: int = 5,
        interp_cache_guard_force_below_zeta: float = 0.005,
        interp_cache_guard_zeta_trigger: float = 2.0e-4,
        profile_timing: bool = True,
        return_results: bool = False,
        mpi_subprocess: bool = False,
        mpi_subprocess_ranks: int = 0,
    ) -> GCEResult | None:
        if input_time is not None and infall_time is not None:
            raise ValueError("Provide only one of input_time or infall_time")
        raw_input_time = input_time if input_time is not None else infall_time
        input_time_arr: np.ndarray | None = None
        infall_values_arr: np.ndarray | None = None
        rhosfr_values_arr: np.ndarray | None = None
        endoftime_eff = int(endoftime)
        if raw_input_time is None and (infall_values is not None or rhosfr_values is not None):
            raise ValueError("input_time (or legacy infall_time) is required when infall_values or rhosfr_values are provided")
        if raw_input_time is not None:
            input_time_arr = np.asarray(raw_input_time, dtype=float).reshape(-1)
            if input_time_arr.size < 2:
                raise ValueError("input_time must contain at least 2 points")
            if not np.all(np.diff(input_time_arr) > 0.0):
                raise ValueError("input_time must be strictly increasing")
            if not np.all(np.isfinite(input_time_arr)):
                raise ValueError("input_time must be finite")
            endoftime_eff = int(np.ceil(float(input_time_arr[-1])))
            if endoftime_eff < 1:
                raise ValueError("input_time[-1] must be > 0")
            if infall_values is None and rhosfr_values is None:
                raise ValueError("input_time requires at least one of infall_values or rhosfr_values")
        if infall_values is not None:
            infall_values_arr = np.asarray(infall_values, dtype=float).reshape(-1)
            if input_time_arr is None:
                raise ValueError("input_time is required with infall_values")
            if input_time_arr.size != infall_values_arr.size:
                raise ValueError("input_time and infall_values must have the same length")
            if not np.all(np.isfinite(infall_values_arr)):
                raise ValueError("infall_values must be finite")
            if np.any(infall_values_arr < 0.0):
                raise ValueError("infall_values must be >= 0")
        if rhosfr_values is not None:
            rhosfr_values_arr = np.asarray(rhosfr_values, dtype=float).reshape(-1)
            if input_time_arr is None:
                raise ValueError("input_time is required with rhosfr_values")
            if input_time_arr.size != rhosfr_values_arr.size:
                raise ValueError("input_time and rhosfr_values must have the same length")
            if not np.all(np.isfinite(rhosfr_values_arr)):
                raise ValueError("rhosfr_values must be finite")
            if np.any(rhosfr_values_arr < 0.0):
                raise ValueError("rhosfr_values must be >= 0")

        d = _build_runconfig_dict(
            endoftime_eff=endoftime_eff,
            sigmat=sigmat, sigmah=sigmah, psfr=psfr, pwind=pwind,
            delay=delay, time_wind=time_wind,
            input_time_arr=input_time_arr,
            infall_values_arr=infall_values_arr,
            rhosfr_values_arr=rhosfr_values_arr,
            use_mpi=use_mpi,
            mpi_nonblocking_reduce=mpi_nonblocking_reduce,
            show_progress=show_progress, progress_style=progress_style,
            output_dir=output_dir, output_mode=output_mode,
            write_output=write_output,
            df_binary_format=df_binary_format, df_write_csv=df_write_csv,
            backend=backend,
            adaptive_timestep=adaptive_timestep,
            dt_min=dt_min, dt_max=dt_max, dt_rel_tol=dt_rel_tol,
            dt_smooth_alpha=dt_smooth_alpha,
            dt_growth_factor=dt_growth_factor,
            dt_shrink_factor=dt_shrink_factor,
            dt_force_small_below_zeta=dt_force_small_below_zeta,
            dt_force_small_value=dt_force_small_value,
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
            interp_cache_guard_samples=interp_cache_guard_samples,
            interp_cache_guard_force_below_zeta=interp_cache_guard_force_below_zeta,
            interp_cache_guard_zeta_trigger=interp_cache_guard_zeta_trigger,
            profile_timing=profile_timing,
        )

        comm0, rank0, size0 = self._mpi_ctx()
        if mpi_subprocess and use_mpi and size0 == 1:
            call_kwargs = dict(d, return_results=True, mpi_subprocess=False, mpi_subprocess_ranks=0)
            return self._run_mpi_subprocess(mpi_subprocess_ranks=mpi_subprocess_ranks, kwargs=call_kwargs)

        # Convert list-valued sequences to tuples for the frozen RunConfig.
        cfg_d = dict(d)
        cfg_d["infall_time"] = None
        for key in ("input_time", "infall_values", "rhosfr_values"):
            v = cfg_d[key]
            if v is not None:
                cfg_d[key] = tuple(v)
        cfg = RunConfig(**cfg_d)

        nmax = NMAX_DEFAULT
        elem = NUM_ELEMENTS

        self._initialize_from_fortran_tables(lowmassive=1, mm=0)
        backend_key = self._backend_key(cfg)
        interp_backend = "jax" if cfg.backend == "jax_full" else cfg.backend
        if self.interpolator is None or self._interpolator_key != backend_key:
            self.interpolator = build_backend(interp_backend, self.tables, cfg=cfg)
            self._interpolator_key = backend_key

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
            return GCEResult(mod=payload["mod_rows"], fis=payload["fis_rows"])
        return None
