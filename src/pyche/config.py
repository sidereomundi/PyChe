"""Run configuration for GCE simulations."""

from __future__ import annotations

from dataclasses import dataclass
import math

@dataclass(frozen=True)
class RunConfig:
    """Container for user-controlled GCE runtime parameters."""

    endoftime: int
    sigmat: float
    sigmah: float
    psfr: float
    pwind: float
    delay: int
    time_wind: int
    input_time: tuple[float, ...] | None = None
    infall_time: tuple[float, ...] | None = None
    infall_values: tuple[float, ...] | None = None
    rhosfr_values: tuple[float, ...] | None = None
    use_mpi: bool = True
    mpi_nonblocking_reduce: bool = False
    show_progress: bool = True
    progress_style: str = "single"
    output_dir: str | None = None
    output_mode: str = "dataframe"
    write_output: bool = True
    df_binary_format: str = "pickle"
    df_write_csv: bool = False
    backend: str = "auto"
    adaptive_timestep: bool = True
    dt_min: int = 1
    dt_max: int = 10
    dt_rel_tol: float = 0.2
    dt_smooth_alpha: float = 0.3
    dt_growth_factor: float = 1.5
    dt_shrink_factor: float = 0.5
    dt_force_small_below_zeta: float = 1.0e-4
    dt_force_small_value: int = 1
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
        if self.input_time is not None and self.infall_time is not None:
            raise ValueError("Provide only one of input_time or infall_time")
        timeline = self.input_time if self.input_time is not None else self.infall_time
        has_infall = self.infall_values is not None
        has_rhosfr = self.rhosfr_values is not None
        if timeline is None and (has_infall or has_rhosfr):
            raise ValueError("input_time (or legacy infall_time) is required when infall_values or rhosfr_values are provided")
        if timeline is not None and not (has_infall or has_rhosfr):
            raise ValueError("input_time requires at least one of infall_values or rhosfr_values")
        if timeline is not None:
            if len(timeline) < 2:
                raise ValueError("input_time must contain at least 2 points")
            prev_t = float(timeline[0])
            if not math.isfinite(prev_t):
                raise ValueError("input_time values must be finite")
            for i in range(1, len(timeline)):
                t = float(timeline[i])
                if not math.isfinite(t):
                    raise ValueError("input_time values must be finite")
                if t <= prev_t:
                    raise ValueError("input_time must be strictly increasing")
                prev_t = t
        if timeline is not None and has_infall:
            assert self.infall_values is not None
            if len(timeline) != len(self.infall_values):
                raise ValueError("input_time and infall_values must have the same length")
            for v in self.infall_values:
                vf = float(v)
                if not math.isfinite(vf):
                    raise ValueError("infall_values must be finite")
                if vf < 0.0:
                    raise ValueError("infall_values must be >= 0")
        if timeline is not None and has_rhosfr:
            assert self.rhosfr_values is not None
            if len(timeline) != len(self.rhosfr_values):
                raise ValueError("input_time and rhosfr_values must have the same length")
            for v in self.rhosfr_values:
                vf = float(v)
                if not math.isfinite(vf):
                    raise ValueError("rhosfr_values must be finite")
                if vf < 0.0:
                    raise ValueError("rhosfr_values must be >= 0")
        if self.output_dir is not None and str(self.output_dir).strip() == "":
            raise ValueError("output_dir cannot be empty")
        if self.progress_style not in {"auto", "single", "line", "compact", "off"}:
            raise ValueError("progress_style must be one of: auto, single, line, compact, off")
        if self.output_mode not in {"legacy", "dataframe", "both"}:
            raise ValueError("output_mode must be one of: legacy, dataframe, both")
        if self.df_binary_format not in {"pickle", "parquet"}:
            raise ValueError("df_binary_format must be one of: pickle, parquet")
        if self.backend not in {"numpy", "cython", "numba", "jax", "jax_full", "auto"}:
            raise ValueError("backend must be one of: numpy, cython, numba, jax, jax_full, auto")
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
        if self.dt_force_small_below_zeta < 0.0:
            raise ValueError("dt_force_small_below_zeta must be >= 0")
        if self.dt_force_small_value < self.dt_min or self.dt_force_small_value > self.dt_max:
            raise ValueError("dt_force_small_value must be within [dt_min, dt_max]")
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
