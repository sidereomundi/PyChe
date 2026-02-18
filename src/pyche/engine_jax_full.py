"""Experimental pure-JAX GCE engine.

This engine is intentionally separate from ``engine_core`` so users can run an
end-to-end JAX execution path (state evolution in ``jax.lax.scan``) without
affecting the reference numpy/cython implementation.
"""

from __future__ import annotations

import sys
from time import perf_counter
from typing import Any

import numpy as np

from .config import RunConfig
from .constants import (
    FE,
    GAS_FLOOR,
    H,
    HEN,
    KAPPA,
    LI,
    MG,
    O16,
    PRIMORDIAL_H,
    PRIMORDIAL_HE4,
    R_GALACTIC,
    SF_THRESHOLD,
    SIGMA_SUN,
    SPALLA_FEH_OFFSET,
    SPALLA_LOG_CONST,
    SPALLA_SLOPE,
    SUPERF,
)
from .output_io import build_fis_rows, build_mod_rows, write_outputs
from .state import SimulationState


def _build_infall_rate(cfg: RunConfig) -> np.ndarray:
    rate = np.zeros(cfg.endoftime + 1, dtype=np.float64)
    if cfg.endoftime <= 0:
        return rate
    tt = np.arange(1, cfg.endoftime + 1, dtype=np.float64)
    if cfg.input_time is not None and cfg.infall_values is not None:
        tvec = np.asarray(cfg.input_time, dtype=np.float64)
        ivec = np.asarray(cfg.infall_values, dtype=np.float64)
        rate[1:] = np.interp(tt, tvec, ivec)
    elif cfg.sigmat != 0.0:
        coeff = cfg.sigmah * SUPERF / (2.5 * cfg.sigmat)
        rate[1:] = coeff * np.exp(-((tt - cfg.delay) ** 2) / (2.0 * cfg.sigmat**2))
    return rate


def _build_psfr_series(cfg: RunConfig) -> np.ndarray:
    psfr = np.full(cfg.endoftime + 1, float(cfg.psfr), dtype=np.float64)
    if cfg.endoftime <= 0:
        return psfr
    if cfg.input_time is not None and cfg.rhosfr_values is not None:
        tvec = np.asarray(cfg.input_time, dtype=np.float64)
        rvec = np.asarray(cfg.rhosfr_values, dtype=np.float64)
        psfr[1:] = np.interp(np.arange(1, cfg.endoftime + 1, dtype=np.float64), tvec, rvec)
    return psfr


def run_mingce_jax_full(
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
    elem: int = 33,
) -> dict[str, np.ndarray] | None:
    del runtime, mstars, binmax, multi1, tdead, norm, ss2, elem
    comm, rank, size = model._mpi_ctx()
    if cfg.use_mpi and comm is not None and size > 1:
        raise NotImplementedError("backend='jax_full' does not support MPI yet")

    try:
        import jax
        from jax import config as jax_config
        import jax.numpy as jnp
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("backend='jax_full' requires JAX installed") from exc

    jax_config.update("jax_enable_x64", True)
    out_t0 = perf_counter()

    infall_rate_np = _build_infall_rate(cfg)
    psfr_np = _build_psfr_series(cfg)
    infall_rate = jnp.asarray(infall_rate_np, dtype=jnp.float64)
    psfr_series = jnp.asarray(psfr_np, dtype=jnp.float64)
    times = jnp.arange(cfg.endoftime + 1, dtype=jnp.int32)

    recycle_frac = jnp.float64(0.30)
    remn_frac = jnp.float64(0.07)
    hot_frac = jnp.float64(0.03)
    snia_eff = jnp.float64(2.0e-4)
    fe_yield = jnp.float64(1.0e-3)
    o_yield = jnp.float64(6.0e-3)
    mg_yield = jnp.float64(8.0e-4)
    li_yield = jnp.float64(1.0e-9)
    he_yield = jnp.float64(2.5e-2)
    prim_h = jnp.float64(PRIMORDIAL_H)
    prim_he4 = jnp.float64(PRIMORDIAL_HE4)
    gas_floor = jnp.float64(GAS_FLOOR)
    sf_threshold = jnp.float64(SF_THRESHOLD)
    superf = jnp.float64(SUPERF)

    def _step(carry, inp):
        allv, gas, stars, remn, hot, wind, olds, zeta, snia, hen, fe, oxy, mg, li, spalla_prev = carry
        t, infall_t, psfr_t = inp

        sfr = jnp.where(
            (gas / superf > sf_threshold) & (cfg.sigmah != 0.0),
            psfr_t
            * (gas / (superf * cfg.sigmah)) ** KAPPA
            * (cfg.sigmah / SIGMA_SUN) ** (KAPPA - 1.0)
            * (8.0 / R_GALACTIC)
            * (superf / 1000.0)
            * cfg.sigmah,
            0.0,
        )
        formed = jnp.minimum(sfr, jnp.maximum(gas - gas_floor, 0.0))
        returned = recycle_frac * formed
        remn_gain = remn_frac * formed
        hot_gain = hot_frac * formed
        wind_loss = jnp.where((cfg.pwind != 0.0) & (t > cfg.time_wind), cfg.pwind * formed, 0.0)

        gas_next = jnp.maximum(gas + infall_t - formed + returned - remn_gain - hot_gain - wind_loss, gas_floor)
        stars_next = stars + formed - returned
        remn_next = remn + remn_gain
        hot_next = hot + hot_gain
        wind_next = wind + wind_loss
        allv_next = allv + infall_t
        olds_next = olds + jnp.where(t > cfg.delay, 0.01 * formed, 0.0)
        snia_inc = jnp.where(t > cfg.delay, snia_eff * formed, 0.0)
        snia_next = snia + snia_inc

        deplete = jnp.clip((formed + wind_loss + hot_gain + remn_gain) / jnp.maximum(gas, gas_floor), 0.0, 1.0)
        hen_next = jnp.maximum(hen * (1.0 - deplete) + he_yield * formed, gas_floor)
        fe_next = jnp.maximum(fe * (1.0 - deplete) + fe_yield * formed, gas_floor)
        oxy_next = jnp.maximum(oxy * (1.0 - deplete) + o_yield * formed, gas_floor)
        mg_next = jnp.maximum(mg * (1.0 - deplete) + mg_yield * formed, gas_floor)
        li_next = jnp.maximum(li * (1.0 - deplete) + li_yield * formed, gas_floor)

        sum_met = fe_next + oxy_next + mg_next + li_next
        zeta_next = jnp.maximum(sum_met / jnp.maximum(gas_next, gas_floor), gas_floor)
        h_mass = jnp.maximum(gas_next * (prim_h - zeta_next) - hen_next, gas_floor)
        sp_now = jnp.power(
            10.0,
            SPALLA_LOG_CONST
            + SPALLA_SLOPE * (jnp.log10(jnp.maximum(fe_next / h_mass, 1.0e-30)) + SPALLA_FEH_OFFSET)
            + jnp.log10(h_mass),
        )
        li_next = li_next + jnp.maximum(sp_now - spalla_prev, 0.0)
        he4 = jnp.maximum(gas_next * prim_he4 + hen_next, gas_floor)

        out = jnp.array(
            [
                allv_next,
                gas_next,
                stars_next,
                remn_next,
                hot_next,
                wind_next,
                olds_next,
                zeta_next,
                snia_next,
                sfr,
                hen_next,
                fe_next,
                oxy_next,
                mg_next,
                li_next,
                sp_now,
                h_mass,
                he4,
            ],
            dtype=jnp.float64,
        )
        carry_next = (
            allv_next,
            gas_next,
            stars_next,
            remn_next,
            hot_next,
            wind_next,
            olds_next,
            zeta_next,
            snia_next,
            hen_next,
            fe_next,
            oxy_next,
            mg_next,
            li_next,
            sp_now,
        )
        return carry_next, out

    @jax.jit
    def _scan_all(infall_vec, psfr_vec, tvec):
        init = (
            jnp.float64(0.0),
            jnp.float64(0.0),
            jnp.float64(0.0),
            jnp.float64(0.0),
            jnp.float64(0.0),
            jnp.float64(0.0),
            jnp.float64(0.0),
            jnp.float64(gas_floor),
            jnp.float64(0.0),
            jnp.float64(gas_floor),
            jnp.float64(gas_floor),
            jnp.float64(gas_floor),
            jnp.float64(gas_floor),
            jnp.float64(gas_floor),
            jnp.float64(0.0),
        )
        return jax.lax.scan(_step, init, (tvec, infall_vec, psfr_vec))

    if rank == 0 and cfg.show_progress and cfg.endoftime > 0:
        print(f"Progress [------------------------------]   0.00% (0/{cfg.endoftime})", flush=True)
    (_, outputs_dev) = _scan_all(infall_rate[1:], psfr_series[1:], times[1:])
    outputs_dev.block_until_ready()
    if rank == 0 and cfg.show_progress and cfg.endoftime > 0:
        print(f"Progress [##############################] 100.00% ({cfg.endoftime}/{cfg.endoftime})", flush=True)
    out = np.asarray(outputs_dev, dtype=np.float64)

    allv = np.zeros(cfg.endoftime + 2, dtype=np.float64)
    gas = np.zeros(cfg.endoftime + 2, dtype=np.float64)
    stars = np.zeros(cfg.endoftime + 2, dtype=np.float64)
    remn = np.zeros(cfg.endoftime + 2, dtype=np.float64)
    hot = np.zeros(cfg.endoftime + 2, dtype=np.float64)
    wind = np.zeros(cfg.endoftime + 2, dtype=np.float64)
    oldstars = np.zeros(cfg.endoftime + 2, dtype=np.float64)
    zeta = np.zeros(cfg.endoftime + 2, dtype=np.float64)
    snianum = np.zeros(cfg.endoftime + 2, dtype=np.float64)
    sfr_hist = np.zeros(cfg.endoftime + 2, dtype=np.float64)
    spalla = np.zeros(cfg.endoftime + 2, dtype=np.float64)
    qqn = np.full((34, cfg.endoftime + 2), GAS_FLOOR, dtype=np.float64)

    sl = slice(1, cfg.endoftime + 1)
    allv[sl] = out[:, 0]
    gas[sl] = out[:, 1]
    stars[sl] = out[:, 2]
    remn[sl] = out[:, 3]
    hot[sl] = out[:, 4]
    wind[sl] = out[:, 5]
    oldstars[sl] = out[:, 6]
    zeta[sl] = out[:, 7]
    snianum[sl] = out[:, 8]
    sfr_hist[sl] = out[:, 9]
    qqn[HEN, sl] = out[:, 10]
    qqn[FE, sl] = out[:, 11]
    qqn[O16, sl] = out[:, 12]
    qqn[MG, sl] = out[:, 13]
    qqn[LI, sl] = out[:, 14]
    spalla[sl] = out[:, 15]
    qqn[H, sl] = out[:, 16]
    qqn[33, sl] = out[:, 17]

    if rank == 0:
        mod_rows = build_mod_rows(cfg.endoftime, allv, gas, stars, sfr_hist, oldstars, qqn)
        fis_rows = build_fis_rows(cfg.endoftime, allv, gas, stars, remn, hot, zeta, sfr_hist, snianum)
        if cfg.write_output:
            from pathlib import Path

            out_dir = Path(cfg.output_dir) if cfg.output_dir is not None else (Path.cwd() / "RISULTATI2")
            write_outputs(
                out_dir,
                output_mode=cfg.output_mode,
                df_binary_format=cfg.df_binary_format,
                df_write_csv=cfg.df_write_csv,
                mod_rows=mod_rows,
                fis_rows=fis_rows,
            )
        print("GCE full JAX run complete")
        print("ninputyield:", model.tables.ninputyield if getattr(model, "tables", None) is not None else 0)
        print("final gas:", gas[cfg.endoftime] if cfg.endoftime > 0 else gas[0])
        print("final zeta:", zeta[cfg.endoftime] if cfg.endoftime > 0 else zeta[0])
        if cfg.profile_timing:
            print(f"timing profile (s): total={perf_counter() - out_t0:.3f}, engine=jax_full")
        if cfg.show_progress and cfg.endoftime > 0:
            sys.stdout.flush()
        return {"mod_rows": mod_rows, "fis_rows": fis_rows}
    return None
