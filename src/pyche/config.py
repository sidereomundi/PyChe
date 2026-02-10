"""Run configuration for MinGCE simulations."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class RunConfig:
    """Container for user-controlled MinGCE runtime parameters."""

    endoftime: int
    sigmat: float
    sigmah: float
    psfr: float
    pwind: float
    delay: int
    time_wind: int
    use_mpi: bool = True
    mpi_nonblocking_reduce: bool = False
    show_progress: bool = True
    output_dir: str | None = None
    output_mode: str = "legacy"
    write_output: bool = True
    df_binary_format: str = "pickle"
    df_write_csv: bool = False
    backend: str = "auto"
    adaptive_timestep: bool = True
    dt_min: int = 1
    dt_max: int = 5
    dt_rel_tol: float = 0.1
    dt_smooth_alpha: float = 0.3
    dt_growth_factor: float = 1.25
    dt_shrink_factor: float = 0.5
    spalla_stride: int = 4
    spalla_inactive_threshold: float = 1.0e-12
    spalla_lut: bool = True
    spalla_lut_q_points: int = 128
    spalla_lut_d_points: int = 128
    spalla_lut_logq_min: float = -30.0
    spalla_lut_logq_max: float = 2.0
    spalla_lut_logd_min: float = -30.0
    spalla_lut_logd_max: float = 2.0
    interp_cache: bool = False
    interp_cache_mass_points: int = 96
    interp_cache_zeta_points: int = 64
    interp_cache_binmax_points: int = 64
    interp_cache_zeta_max: float = 0.05
    interp_cache_guard: bool = True
    interp_cache_guard_tol: float = 0.05
    interp_cache_guard_stride: int = 16
    interp_cache_guard_samples: int = 5
    interp_cache_guard_force_below_zeta: float = 0.005
    interp_cache_guard_zeta_trigger: float = 2.0e-4
    profile_timing: bool = True

    def __post_init__(self) -> None:
        if self.endoftime < 0:
            raise ValueError("endoftime must be >= 0")
        if self.sigmat < 0.0:
            raise ValueError("sigmat must be >= 0")
        if self.output_dir is not None and str(self.output_dir).strip() == "":
            raise ValueError("output_dir cannot be empty")
        if self.output_mode not in {"legacy", "dataframe", "both"}:
            raise ValueError("output_mode must be one of: legacy, dataframe, both")
        if self.df_binary_format not in {"pickle", "parquet"}:
            raise ValueError("df_binary_format must be one of: pickle, parquet")
        if self.backend not in {"numpy", "cython", "numba", "auto"}:
            raise ValueError("backend must be one of: numpy, cython, numba, auto")
        if self.dt_min < 1:
            raise ValueError("dt_min must be >= 1")
        if self.dt_max < self.dt_min:
            raise ValueError("dt_max must be >= dt_min")
        if self.dt_rel_tol <= 0.0:
            raise ValueError("dt_rel_tol must be > 0")
        if not (0.0 < self.dt_smooth_alpha <= 1.0):
            raise ValueError("dt_smooth_alpha must be in (0, 1]")
        if self.dt_growth_factor < 1.0:
            raise ValueError("dt_growth_factor must be >= 1")
        if not (0.0 < self.dt_shrink_factor <= 1.0):
            raise ValueError("dt_shrink_factor must be in (0, 1]")
        if self.spalla_stride < 1:
            raise ValueError("spalla_stride must be >= 1")
        if self.spalla_inactive_threshold < 0.0:
            raise ValueError("spalla_inactive_threshold must be >= 0")
        if self.spalla_lut_q_points < 2:
            raise ValueError("spalla_lut_q_points must be >= 2")
        if self.spalla_lut_d_points < 2:
            raise ValueError("spalla_lut_d_points must be >= 2")
        if self.spalla_lut_logq_max <= self.spalla_lut_logq_min:
            raise ValueError("spalla_lut_logq_max must be > spalla_lut_logq_min")
        if self.spalla_lut_logd_max <= self.spalla_lut_logd_min:
            raise ValueError("spalla_lut_logd_max must be > spalla_lut_logd_min")
        if self.interp_cache_mass_points < 2:
            raise ValueError("interp_cache_mass_points must be >= 2")
        if self.interp_cache_zeta_points < 2:
            raise ValueError("interp_cache_zeta_points must be >= 2")
        if self.interp_cache_binmax_points < 2:
            raise ValueError("interp_cache_binmax_points must be >= 2")
        if self.interp_cache_zeta_max <= 0.0:
            raise ValueError("interp_cache_zeta_max must be > 0")
        if self.interp_cache_guard_tol <= 0.0:
            raise ValueError("interp_cache_guard_tol must be > 0")
        if self.interp_cache_guard_stride < 1:
            raise ValueError("interp_cache_guard_stride must be >= 1")
        if self.interp_cache_guard_samples < 1:
            raise ValueError("interp_cache_guard_samples must be >= 1")
        if self.interp_cache_guard_force_below_zeta < 0.0:
            raise ValueError("interp_cache_guard_force_below_zeta must be >= 0")
        if self.interp_cache_guard_zeta_trigger < 0.0:
            raise ValueError("interp_cache_guard_zeta_trigger must be >= 0")
