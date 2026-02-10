"""Shared MinGCE runtime loop used by serial and MPI engine frontends."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .config import RunConfig
from .output_io import build_fis_rows, build_mod_rows, write_outputs
from .state import SimulationState

try:
    from . import _cyengine  # type: ignore
except Exception:  # pragma: no cover - optional compiled extension
    _cyengine = None


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

    superf = 20000.0
    threshold = 0.1
    sigmasun = 50.0
    kappa = 1.5
    rm = 8.0

    out_dir = Path(cfg.output_dir) if cfg.output_dir is not None else (Path.cwd() / "RISULTATI2")

    elem_idx_no14 = np.array([i for i in range(1, elem) if i != 14], dtype=int)
    wind_idx = np.array([i for i in range(1, 32) if i != 14], dtype=int)
    zeta_idx = np.array([i for i in range(2, elem - 1) if i != 14], dtype=int)
    t = 0
    last_progress_step = 0
    progress_stride = max(1, cfg.endoftime // 200) if cfg.endoftime > 0 else 1
    progress_use_carriage = bool(getattr(sys.stdout, "isatty", lambda: False)())
    progress_path = os.getenv("PYCHE_PROGRESS_PATH", "").strip() if rank == 0 else ""
    gas_floor = 1.0e-20
    stage_sec = {"interp": 0.0, "mpi_reduce": 0.0, "death": 0.0, "wind": 0.0, "output": 0.0}
    total_t0 = perf_counter()
    rel_ema = 0.0
    rel_ema_init = False
    stable_steps = 0

    # Static infall profile: allv depends only on (sigmat, sigmah, delay), not on the evolving state arrays.
    if cfg.endoftime > 0:
        tt = np.arange(1, cfg.endoftime + 1, dtype=float)
        if cfg.sigmat != 0.0:
            coeff = cfg.sigmah * superf / (2.5 * cfg.sigmat)
            infall = coeff * np.exp(-((tt - cfg.delay) ** 2) / (2.0 * cfg.sigmat**2))
            allv[1 : cfg.endoftime + 1] = np.cumsum(infall, dtype=float)
        else:
            allv[1 : cfg.endoftime + 1] = allv[0]

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
    use_cyengine = bool(cfg.backend in {"cython", "auto"} and _cyengine is not None)
    use_spalla_lut = bool(use_cyengine and cfg.spalla_lut)
    if use_spalla_lut:
        lut_logq = np.linspace(cfg.spalla_lut_logq_min, cfg.spalla_lut_logq_max, cfg.spalla_lut_q_points, dtype=float)
        lut_logd = np.linspace(cfg.spalla_lut_logd_min, cfg.spalla_lut_logd_max, cfg.spalla_lut_d_points, dtype=float)
        logq_grid = lut_logq[:, None]
        logd_grid = lut_logd[None, :]
        spalla_lut = np.power(10.0, -9.50 + 1.24 * (logq_grid - logd_grid + 2.75) + logd_grid).astype(float, copy=False)
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
        if progress_path:
            try:
                with open(progress_path, "w", encoding="utf-8") as f:
                    f.write(str(int(step)))
            except OSError:
                pass
        if not (cfg.show_progress and rank == 0 and cfg.endoftime > 0):
            return
        pct = 100.0 * step / cfg.endoftime
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

        if gas[t] / superf > threshold and cfg.sigmah != 0.0:
            sfr = (
                cfg.psfr
                * (gas[t] / (superf * cfg.sigmah)) ** kappa
                * (cfg.sigmah / sigmasun) ** (kappa - 1.0)
                * (8.0 / rm)
                * (superf / 1000.0)
                * cfg.sigmah
            )
        else:
            sfr = 0.0
        sfr_hist[t] = sfr
        if dt_scale > 1:
            sfr_hist[t_prev + 1 : t] = sfr
        sfr_mass = sfr * float(dt_scale)

        if gas[t] / superf >= threshold:
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
                    if tdead[jj] + t > 13500.0:
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
                        qqn[31, t3] = (
                            qqn[31, t3] - qqn[31, t] * (mstars1_eff[jj] + hecores[jj]) * multi1[jj] * sfr_mass / gas[t]
                        )

                    if died_now:
                        if next_died:
                            jj += 1
                            if jj > ss2:
                                break
                            continue
                        stars[t3] += starstot
                        q14 = qacc[14]
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
                        q14 = qacc[14]
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
                    gas_floor,
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
                    gas_floor,
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
            while True:
                if windist != 0.0:
                    wind[t3] += windist
                if wind_scale != 0.0:
                    qqn[wind_idx, t3] = qqn[wind_idx, t3] - qqn[wind_idx, t] * winds[wind_idx] * wind_scale
                    np.maximum(qqn[wind_idx, t3], gas_floor, out=qqn[wind_idx, t3])

                zeta[t3] = (
                    float(np.sum(qqn[zeta_idx, t3])) / gas[t3] if (sfr > 0.0 and gas[t3] > 0.0) else gas_floor
                )

                qqn[elem, t3] = gas[t3] * 0.241 + qqn[1, t3]
                qqn[elem - 1, t3] = gas[t3] * (0.759 - zeta[t3]) - qqn[1, t3]

                if t > 1:
                    qqn[31, t3] = qqn[31, t3] + (allv[t] - allv[max(0, t - dt_scale)]) * ini[31]
                    denom = max(qqn[elem - 1, t3], 1.0e-30)
                    feh = np.log10(max(qqn[9, t3] / denom, 1.0e-30))
                    spalla[t3] = 10 ** (-9.50 + 1.24 * (feh - (-2.75)) + np.log10(denom))
                    qqn[31, t3] = qqn[31, t3] + spalla[t3] - spalla[t3 - 1]
                else:
                    qqn[31, t3] = qqn[31, t3] + allv[t] * ini[31]

                if t3 >= cfg.endoftime:
                    break
                t3 += 1
        stage_sec["wind"] += perf_counter() - wind_t0

        if t - last_progress_step >= progress_stride:
            _print_progress(t)
            last_progress_step = t
        if cfg.adaptive_timestep and t < cfg.endoftime:
            dg = abs(gas[t] - gas[t_prev]) / max(abs(gas[t_prev]), 1.0)
            dz = abs(zeta[t] - zeta[t_prev]) / max(abs(zeta[t_prev]), gas_floor)
            rel = max(dg, dz)
            if not rel_ema_init:
                rel_ema = rel
                rel_ema_init = True
            else:
                rel_ema = (1.0 - cfg.dt_smooth_alpha) * rel_ema + cfg.dt_smooth_alpha * rel
            if sfr <= 0.0 and gas[t] / superf < threshold * 1.05:
                next_dt = cfg.dt_max
                stable_steps = 0
            elif rel_ema <= 0.0:
                next_dt = cfg.dt_max
                stable_steps = 0
            else:
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
                next_dt = proposed

    if rank == 0:
        if cfg.show_progress and cfg.endoftime > 0:
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
        print("MinGCE full translation run complete")
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
