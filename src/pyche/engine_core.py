"""Shared GCE runtime loop used by serial and MPI engine frontends."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .config import RunConfig
from .constants import (
    FE,
    GALACTIC_AGE,
    GAS_FLOOR,
    HEN,
    KAPPA,
    LI,
    PRIMORDIAL_H,
    PRIMORDIAL_HE4,
    R_GALACTIC,
    REMN,
    SF_THRESHOLD,
    SIGMA_SUN,
    SPALLA_FEH_OFFSET,
    SPALLA_LOG_CONST,
    SPALLA_SLOPE,
    SUPERF,
)
from .output_io import build_fis_rows, build_mod_rows, write_outputs
from .state import SimulationState

try:
    from . import _cyengine  # type: ignore
except Exception:  # pragma: no cover - optional compiled extension
    _cyengine = None


def _compute_static_infall(
    cfg: RunConfig,
    allv: np.ndarray,
    input_time_arr: np.ndarray | None,
    infall_values_arr: np.ndarray | None,
) -> None:
    """Pre-compute cumulative infall profile into *allv*."""
    if cfg.endoftime <= 0:
        return
    tt = np.arange(1, cfg.endoftime + 1, dtype=float)
    if input_time_arr is not None and infall_values_arr is not None:
        infall = np.interp(tt, input_time_arr, infall_values_arr)
        allv[1 : cfg.endoftime + 1] = np.cumsum(infall, dtype=float)
    elif cfg.sigmat != 0.0:
        coeff = cfg.sigmah * SUPERF / (2.5 * cfg.sigmat)
        infall = coeff * np.exp(-((tt - cfg.delay) ** 2) / (2.0 * cfg.sigmat**2))
        allv[1 : cfg.endoftime + 1] = np.cumsum(infall, dtype=float)
    else:
        allv[1 : cfg.endoftime + 1] = allv[0]


def _compute_sfr(
    cfg: RunConfig,
    gas_t: float,
    psfr_eff: float,
) -> float:
    """Star formation rate at the current timestep."""
    if gas_t / SUPERF > SF_THRESHOLD and cfg.sigmah != 0.0:
        return (
            psfr_eff
            * (gas_t / (SUPERF * cfg.sigmah)) ** KAPPA
            * (cfg.sigmah / SIGMA_SUN) ** (KAPPA - 1.0)
            * (8.0 / R_GALACTIC)
            * (SUPERF / 1000.0)
            * cfg.sigmah
        )
    return 0.0


def _wind_chemistry_step(
    t: int,
    cfg: RunConfig,
    dt_scale: int,
    sfr: float,
    windist: float,
    wind_scale: float,
    elem: int,
    wind_idx: np.ndarray,
    zeta_idx: np.ndarray,
    gas: np.ndarray,
    allv: np.ndarray,
    ini: np.ndarray,
    winds: np.ndarray,
    qqn: np.ndarray,
    wind: np.ndarray,
    zeta: np.ndarray,
    spalla: np.ndarray,
) -> None:
    """Wind processing + metallicity + spallation (pure-Python path)."""
    t3 = t
    while True:
        if windist != 0.0:
            wind[t3] += windist
        if wind_scale != 0.0:
            qqn[wind_idx, t3] = qqn[wind_idx, t3] - qqn[wind_idx, t] * winds[wind_idx] * wind_scale
            np.maximum(qqn[wind_idx, t3], GAS_FLOOR, out=qqn[wind_idx, t3])

        zeta[t3] = (
            float(np.sum(qqn[zeta_idx, t3])) / gas[t3] if (sfr > 0.0 and gas[t3] > 0.0) else GAS_FLOOR
        )

        qqn[elem, t3] = gas[t3] * PRIMORDIAL_HE4 + qqn[HEN, t3]
        qqn[elem - 1, t3] = gas[t3] * (PRIMORDIAL_H - zeta[t3]) - qqn[HEN, t3]

        if t > 1:
            qqn[LI, t3] = qqn[LI, t3] + (allv[t] - allv[max(0, t - dt_scale)]) * ini[LI]
            denom = max(qqn[elem - 1, t3], 1.0e-30)
            feh = np.log10(max(qqn[FE, t3] / denom, 1.0e-30))
            spalla[t3] = 10 ** (SPALLA_LOG_CONST + SPALLA_SLOPE * (feh - (-SPALLA_FEH_OFFSET)) + np.log10(denom))
            qqn[LI, t3] = qqn[LI, t3] + spalla[t3] - spalla[t3 - 1]
        else:
            qqn[LI, t3] = qqn[LI, t3] + allv[t] * ini[LI]

        if t3 >= cfg.endoftime:
            break
        t3 += 1


def _adapt_timestep(
    cfg: RunConfig,
    t: int,
    t_prev: int,
    gas: np.ndarray,
    zeta: np.ndarray,
    sfr: float,
    rel_ema: float,
    rel_ema_init: bool,
    stable_steps: int,
) -> tuple[int, float, bool, int]:
    """Adaptive timestep calculation. Returns (next_dt, rel_ema, rel_ema_init, stable_steps)."""
    dg = abs(gas[t] - gas[t_prev]) / max(abs(gas[t_prev]), 1.0)
    dz = abs(zeta[t] - zeta[t_prev]) / max(abs(zeta[t_prev]), GAS_FLOOR)
    rel = max(dg, dz)
    if not rel_ema_init:
        rel_ema = rel
        rel_ema_init = True
    else:
        rel_ema = (1.0 - cfg.dt_smooth_alpha) * rel_ema + cfg.dt_smooth_alpha * rel
    if sfr <= 0.0 and gas[t] / SUPERF < SF_THRESHOLD * 1.05:
        return cfg.dt_max, rel_ema, rel_ema_init, 0
    if rel_ema <= 0.0:
        return cfg.dt_max, rel_ema, rel_ema_init, 0
    proposed = int(np.clip(cfg.dt_rel_tol / max(rel_ema, 1.0e-12), cfg.dt_min, cfg.dt_max))
    if rel > cfg.dt_rel_tol * 1.5:
        proposed = max(cfg.dt_min, int(max(1, np.floor(proposed * cfg.dt_shrink_factor))))
        stable_steps = 0
    else:
        if rel_ema < cfg.dt_rel_tol * 0.4:
            stable_steps += 1
        else:
            stable_steps = 0
        if stable_steps >= 3:
            grown = int(np.ceil(max(1.0, proposed) * cfg.dt_growth_factor))
            proposed = min(cfg.dt_max, max(proposed, grown))
            stable_steps = 0
    if zeta[t] < cfg.dt_force_small_below_zeta:
        proposed = min(proposed, cfg.dt_force_small_value)
    return proposed, rel_ema, rel_ema_init, stable_steps


def run_mingce_loop(
    model: Any,
    cfg: RunConfig,
    runtime: SimulationState,
    *,
    mstars: np.ndarray,
    binmax: np.ndarray,
    multi1: np.ndarray,
    tdead: np.ndarray,
    norm: float,
    ss2: int,
    elem: int,
    comm: object | None,
    rank: int,
    size: int,
    mpi_active: bool,
    mpi_module: object | None,
) -> dict[str, np.ndarray] | None:
    allv = runtime.allv
    gas = runtime.gas
    stars = runtime.stars
    remn = runtime.remn
    hot = runtime.hot
    wind = runtime.wind
    oldstars = runtime.oldstars
    zeta = runtime.zeta
    snianum = runtime.snianum
    spalla = runtime.spalla
    sfr_hist = runtime.sfr_hist
    qqn = runtime.qqn
    ini = runtime.ini
    winds = runtime.winds

    out_dir = Path(cfg.output_dir) if cfg.output_dir is not None else (Path.cwd() / "RISULTATI2")

    elem_idx_no14 = np.array([i for i in range(1, elem) if i != REMN], dtype=int)
    wind_idx = np.array([i for i in range(1, 32) if i != REMN], dtype=int)
    zeta_idx = np.array([i for i in range(2, elem - 1) if i != REMN], dtype=int)
    t = 0
    last_progress_step = 0
    progress_stride = max(1, cfg.endoftime // 200) if cfg.endoftime > 0 else 1
    style = str(getattr(cfg, "progress_style", "single")).strip().lower()
    if style not in {"auto", "single", "line", "compact", "off"}:
        style = "auto"
    if style == "auto":
        style = "single" if bool(getattr(sys.stdout, "isatty", lambda: False)()) else "line"
    progress_use_carriage = style == "single"
    progress_path = os.getenv("PYCHE_PROGRESS_PATH", "").strip() if rank == 0 else ""
    last_progress_bucket = -1
    stage_sec = {"interp": 0.0, "mpi_reduce": 0.0, "death": 0.0, "wind": 0.0, "output": 0.0}
    total_t0 = perf_counter()
    rel_ema = 0.0
    rel_ema_init = False
    stable_steps = 0

    input_time_arr: np.ndarray | None = None
    if cfg.input_time is not None:
        input_time_arr = np.asarray(cfg.input_time, dtype=float)
    elif cfg.infall_time is not None:
        input_time_arr = np.asarray(cfg.infall_time, dtype=float)

    infall_values_arr: np.ndarray | None = None
    if cfg.infall_values is not None:
        infall_values_arr = np.asarray(cfg.infall_values, dtype=float)

    rhosfr_values_arr: np.ndarray | None = None
    if cfg.rhosfr_values is not None:
        rhosfr_values_arr = np.asarray(cfg.rhosfr_values, dtype=float)

    _compute_static_infall(cfg, allv, input_time_arr, infall_values_arr)

    # Reuse these buffers every timestep to avoid heavy allocator churn.
    hecores = np.zeros(ss2 + 2, dtype=float)
    mstars1_eff = np.where(binmax > 0.0, binmax + mstars, mstars).astype(float, copy=False)
    qispecial = np.zeros((elem + 1, ss2 + 2), dtype=float)
    qacc = np.zeros(elem + 1, dtype=float)
    local_bins = range(1 + rank, ss2 + 1, size) if mpi_active else range(1, ss2 + 1)
    local_bins_arr = np.asarray(list(local_bins), dtype=np.int32)
    interp = model.interpolator.interp
    interp_many_step = getattr(model.interpolator, "interp_many_step", None)
    next_dt = 1
    use_cyengine = bool(cfg.backend in {"cython", "auto", "jax"} and _cyengine is not None)
    use_spalla_lut = bool(use_cyengine and cfg.spalla_lut)
    if use_spalla_lut:
        lut_logq = np.linspace(cfg.spalla_lut_logq_min, cfg.spalla_lut_logq_max, cfg.spalla_lut_q_points, dtype=float)
        lut_logd = np.linspace(cfg.spalla_lut_logd_min, cfg.spalla_lut_logd_max, cfg.spalla_lut_d_points, dtype=float)
        logq_grid = lut_logq[:, None]
        logd_grid = lut_logd[None, :]
        spalla_lut = np.power(10.0, SPALLA_LOG_CONST + SPALLA_SLOPE * (logq_grid - logd_grid + SPALLA_FEH_OFFSET) + logd_grid).astype(float, copy=False)
    else:
        spalla_lut = np.empty((0, 0), dtype=float)
    mpi_reduce_buf = None
    mpi_q_view = None
    old_slice = slice(0, 1)
    he_slice = qi_slice = slice(0, 0)
    if mpi_active and comm is not None and mpi_module is not None:
        he_n = hecores.size
        qi_n = qispecial.size
        off = 1
        he_slice = slice(off, off + he_n)
        off += he_n
        qi_slice = slice(off, off + qi_n)
        mpi_reduce_buf = np.zeros(off + qi_n, dtype=float)
        mpi_q_view = mpi_reduce_buf[qi_slice].reshape(qispecial.shape)

    def _print_progress(step: int) -> None:
        nonlocal last_progress_bucket
        if progress_path:
            try:
                with open(progress_path, "w", encoding="utf-8") as f:
                    f.write(str(int(step)))
            except OSError:
                pass
        if not (cfg.show_progress and rank == 0 and cfg.endoftime > 0):
            return
        if style == "off":
            return
        pct = 100.0 * step / cfg.endoftime
        if style == "compact":
            bucket = int(pct) // 5
            if bucket == last_progress_bucket and step < cfg.endoftime:
                return
            last_progress_bucket = bucket
        bar_w = 30
        fill = int(bar_w * step / cfg.endoftime)
        bar = "#" * fill + "-" * (bar_w - fill)
        msg = f"Progress [{bar}] {pct:6.2f}% ({step}/{cfg.endoftime})"
        if progress_use_carriage:
            sys.stdout.write(f"\r{msg}")
            sys.stdout.flush()
        else:
            print(msg, flush=True)

    while t < cfg.endoftime:
        if cfg.adaptive_timestep:
            step_dt = max(cfg.dt_min, min(cfg.dt_max, int(next_dt)))
            if zeta[t] < cfg.dt_force_small_below_zeta:
                step_dt = min(step_dt, cfg.dt_force_small_value)
        else:
            step_dt = 1
        t_prev = t
        t = min(cfg.endoftime, t + step_dt)
        dt_scale = max(1, t - t_prev)

        # Keep gas consistent with the current state before the generation formed at time t is injected.
        gas[t : cfg.endoftime + 1] = (
            allv[t : cfg.endoftime + 1]
            - stars[t : cfg.endoftime + 1]
            - remn[t : cfg.endoftime + 1]
            - hot[t : cfg.endoftime + 1]
            - wind[t : cfg.endoftime + 1]
        )

        psfr_eff = cfg.psfr
        if input_time_arr is not None and rhosfr_values_arr is not None:
            psfr_eff = float(np.interp(float(t), input_time_arr, rhosfr_values_arr))
        sfr = _compute_sfr(cfg, float(gas[t]), psfr_eff)
        sfr_hist[t] = sfr
        if dt_scale > 1:
            sfr_hist[t_prev + 1 : t] = sfr
        sfr_mass = sfr * float(dt_scale)

        if gas[t] / SUPERF >= SF_THRESHOLD:
            interp_t0 = perf_counter()
            hecores.fill(0.0)
            qispecial.fill(0.0)
            oldstars_contrib = 0.0

            if interp_many_step is not None:
                oldstars_contrib = float(
                    interp_many_step(
                        t=t,
                        sfr_mass=float(sfr_mass),
                        zeta_t=float(zeta[t]),
                        elem=elem,
                        indices=local_bins_arr,
                        mstars=mstars,
                        binmax=binmax,
                        multi1=multi1,
                        tdead=tdead,
                        hecores=hecores,
                        mstars1_eff=mstars1_eff,
                        qispecial=qispecial,
                    )
                )
            else:
                for jj in local_bins:
                    if tdead[jj] + t > GALACTIC_AGE:
                        oldstars_contrib += multi1[jj] * sfr_mass

                    q, hecore = interp(mstars[jj], zeta[t], binmax[jj])
                    hecores[jj] = hecore
                    qispecial[1:elem, jj] = q[: elem - 1]
            stage_sec["interp"] += perf_counter() - interp_t0

            if mpi_active and comm is not None and mpi_module is not None:
                mpi_t0 = perf_counter()
                assert mpi_reduce_buf is not None
                assert mpi_q_view is not None
                mpi_reduce_buf[old_slice] = oldstars_contrib
                mpi_reduce_buf[he_slice] = hecores
                mpi_q_view[:, :] = qispecial
                if cfg.mpi_nonblocking_reduce:
                    req = comm.Iallreduce(mpi_module.IN_PLACE, mpi_reduce_buf, op=mpi_module.SUM)
                    req.Wait()
                else:
                    comm.Allreduce(mpi_module.IN_PLACE, mpi_reduce_buf, op=mpi_module.SUM)
                oldstars[t] += float(mpi_reduce_buf[0])
                hecores[:] = mpi_reduce_buf[he_slice]
                qispecial[:, :] = mpi_q_view
                stage_sec["mpi_reduce"] += perf_counter() - mpi_t0
            else:
                oldstars[t] += oldstars_contrib

            # Death/enrichment block (hot path -- stays inline; Cython counterpart exists).
            t3 = t
            starstot = sfr_mass * norm
            difftot = sfr_mass * norm
            snian = 0.0
            jj = 1
            qacc.fill(0.0)
            death_t0 = perf_counter()
            if use_cyengine:
                _cyengine.death_enrichment_step(
                    t,
                    cfg.endoftime,
                    ss2,
                    elem,
                    float(gas[t]),
                    float(sfr_mass),
                    float(norm),
                    tdead,
                    multi1,
                    binmax,
                    mstars1_eff,
                    hecores,
                    qispecial,
                    qacc,
                    qqn,
                    stars,
                    remn,
                    snianum,
                )
            else:
                while True:
                    next_dead = tdead[jj + 1] if (jj + 1) <= ss2 else 1.0e30
                    died_now = t3 >= (t + tdead[jj])
                    next_died = t3 >= (t + next_dead)

                    if died_now:
                        dm = multi1[jj] * sfr_mass
                        qacc[1:elem] += qispecial[1:elem, jj] * dm
                        difftot -= (mstars1_eff[jj] - hecores[jj]) * dm
                        starstot -= mstars1_eff[jj] * dm
                        if binmax[jj] > 0.0:
                            snian += dm

                    if (not next_died) and gas[t] > 0.0:
                        qqn[elem_idx_no14, t3] = (
                            qqn[elem_idx_no14, t3] + qacc[elem_idx_no14] - qqn[elem_idx_no14, t] * difftot / gas[t]
                        )
                        qqn[LI, t3] = (
                            qqn[LI, t3] - qqn[LI, t] * (mstars1_eff[jj] + hecores[jj]) * multi1[jj] * sfr_mass / gas[t]
                        )

                    if died_now:
                        if next_died:
                            jj += 1
                            if jj > ss2:
                                break
                            continue
                        stars[t3] += starstot
                        q14 = qacc[REMN]
                        remn[t3] += q14
                        if q14 < 0.0:
                            break
                        snianum[t3] += snian
                        if t3 >= cfg.endoftime:
                            break
                        t3 += 1
                        jj += 1
                        if jj > ss2:
                            break
                    else:
                        stars[t3] += starstot
                        q14 = qacc[REMN]
                        remn[t3] += q14
                        if q14 < 0.0:
                            break
                        snianum[t3] += snian
                        if t3 >= cfg.endoftime:
                            break
                        t3 += 1
            stage_sec["death"] += perf_counter() - death_t0

        t3 = t
        windist = cfg.pwind * sfr_mass if (cfg.pwind != 0.0 and t > cfg.time_wind and gas[t] > 0.0) else 0.0
        gas_t = gas[t]
        wind_scale = (windist / gas_t) if (sfr > 0.0 and gas_t > 0.0) else 0.0

        wind_t0 = perf_counter()
        if use_cyengine:
            if wind_scale == 0.0:
                _cyengine.wind_chem_step_no_wind(
                    t,
                    cfg.endoftime,
                    elem,
                    dt_scale,
                    float(sfr),
                    GAS_FLOOR,
                    float(windist),
                    float(wind_scale),
                    gas,
                    allv,
                    ini,
                    qqn,
                    wind,
                    zeta,
                    spalla,
                    int(cfg.spalla_stride),
                    float(cfg.spalla_inactive_threshold),
                    bool(use_spalla_lut),
                    spalla_lut,
                    float(cfg.spalla_lut_logq_min),
                    float(cfg.spalla_lut_logq_max),
                    float(cfg.spalla_lut_logd_min),
                    float(cfg.spalla_lut_logd_max),
                )
            else:
                _cyengine.wind_chem_step_with_wind(
                    t,
                    cfg.endoftime,
                    elem,
                    dt_scale,
                    float(sfr),
                    GAS_FLOOR,
                    float(windist),
                    float(wind_scale),
                    gas,
                    allv,
                    ini,
                    winds,
                    qqn,
                    wind,
                    zeta,
                    spalla,
                    int(cfg.spalla_stride),
                    float(cfg.spalla_inactive_threshold),
                    bool(use_spalla_lut),
                    spalla_lut,
                    float(cfg.spalla_lut_logq_min),
                    float(cfg.spalla_lut_logq_max),
                    float(cfg.spalla_lut_logd_min),
                    float(cfg.spalla_lut_logd_max),
                )
        else:
            _wind_chemistry_step(
                t, cfg, dt_scale, sfr, windist, wind_scale, elem,
                wind_idx, zeta_idx, gas, allv, ini, winds, qqn, wind, zeta, spalla,
            )
        stage_sec["wind"] += perf_counter() - wind_t0

        if t - last_progress_step >= progress_stride:
            _print_progress(t)
            last_progress_step = t
        if cfg.adaptive_timestep and t < cfg.endoftime:
            next_dt, rel_ema, rel_ema_init, stable_steps = _adapt_timestep(
                cfg, t, t_prev, gas, zeta, sfr, rel_ema, rel_ema_init, stable_steps,
            )

    if rank == 0:
        if cfg.show_progress and cfg.endoftime > 0:
            if last_progress_step < cfg.endoftime:
                _print_progress(cfg.endoftime)
            if progress_use_carriage:
                sys.stdout.write("\n")
                sys.stdout.flush()

        out_t0 = perf_counter()
        mod_rows = build_mod_rows(cfg.endoftime, allv, gas, stars, sfr_hist, oldstars, qqn)
        fis_rows = build_fis_rows(cfg.endoftime, allv, gas, stars, remn, hot, zeta, sfr_hist, snianum)
        if cfg.write_output:
            write_outputs(
                out_dir,
                output_mode=cfg.output_mode,
                df_binary_format=cfg.df_binary_format,
                df_write_csv=cfg.df_write_csv,
                mod_rows=mod_rows,
                fis_rows=fis_rows,
            )
            stage_sec["output"] += perf_counter() - out_t0
        else:
            stage_sec["output"] += perf_counter() - out_t0
        print("GCE full translation run complete")
        if mpi_active:
            print("MPI ranks:", size)
        ninput = model.tables.ninputyield if getattr(model, "tables", None) is not None else model.state.ninputyield
        print("ninputyield:", ninput)
        print("final gas:", gas[cfg.endoftime] if cfg.endoftime > 0 else gas[0])
        print("final zeta:", zeta[cfg.endoftime] if cfg.endoftime > 0 else zeta[0])
        if cfg.profile_timing:
            total = perf_counter() - total_t0
            known = stage_sec["interp"] + stage_sec["mpi_reduce"] + stage_sec["death"] + stage_sec["wind"] + stage_sec["output"]
            other = max(0.0, total - known)
            print(
                "timing profile (s): "
                f"total={total:.3f}, interp={stage_sec['interp']:.3f}, mpi_reduce={stage_sec['mpi_reduce']:.3f}, "
                f"death={stage_sec['death']:.3f}, wind={stage_sec['wind']:.3f}, output={stage_sec['output']:.3f}, other={other:.3f}"
            )
        return {"mod_rows": mod_rows, "fis_rows": fis_rows}
    return None
