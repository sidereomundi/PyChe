"""Utilities to validate interpolation-cache guard settings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from .main import GCEModel


@dataclass
class CacheValidationResult:
    n_points: int
    ofe_max_abs_diff: float
    ofe_p95_abs_diff: float
    ofe_mean_abs_diff: float
    feh_max_abs_diff: float
    feh_p95_abs_diff: float
    feh_mean_abs_diff: float


def _abundance_series(res):
    cols = {n: i for i, n in enumerate(res.mod_columns)}
    fe = res.mod[:, cols["Fe"]]
    h = res.mod[:, cols["H"]]
    o16 = res.mod[:, cols["O16"]]
    eps = 1.0e-30
    fe_h_sun_mass = 56.0 * 10.0 ** (-4.50)
    o_fe_sun_mass = (16.0 / 56.0) * 10.0 ** (8.69 - 7.50)
    log_feh = np.log10(np.maximum(fe / np.maximum(h, eps), eps) / fe_h_sun_mass)
    log_o_fe = np.log10(np.maximum(o16 / np.maximum(fe, eps), eps) / o_fe_sun_mass)
    return log_feh, log_o_fe


def validate_cache_stride(
    *,
    stride_test: int,
    endoftime: int = 2000,
    sigmat: float = 3000.0,
    sigmah: float = 50.0,
    psfr: float = 0.3,
    pwind: float = 0.0,
    delay: int = 10000,
    time_wind: int = 10000,
    tol: float = 0.01,
    samples: int = 11,
    mass_points: int = 160,
    zeta_points: int = 128,
    binmax_points: int = 128,
) -> CacheValidationResult:
    if stride_test < 1:
        raise ValueError("stride_test must be >= 1")

    base = dict(
        endoftime=endoftime,
        sigmat=sigmat,
        sigmah=sigmah,
        psfr=psfr,
        pwind=pwind,
        delay=delay,
        time_wind=time_wind,
        use_mpi=False,
        show_progress=False,
        backend="auto",
        output_mode="dataframe",
        write_output=False,
        return_results=True,
        adaptive_timestep=True,
        dt_min=1,
        dt_max=5,
        dt_rel_tol=0.1,
        dt_smooth_alpha=0.3,
        dt_growth_factor=1.25,
        dt_shrink_factor=0.5,
        interp_cache=True,
        interp_cache_mass_points=mass_points,
        interp_cache_zeta_points=zeta_points,
        interp_cache_binmax_points=binmax_points,
        interp_cache_guard=True,
        interp_cache_guard_tol=tol,
        interp_cache_guard_samples=samples,
        spalla_stride=4,
        spalla_inactive_threshold=1.0e-12,
        spalla_lut=True,
    )

    ref = GCEModel().GCE(**base, interp_cache_guard_stride=1)
    test = GCEModel().GCE(**base, interp_cache_guard_stride=stride_test)
    feh_ref, ofe_ref = _abundance_series(ref)
    feh_test, ofe_test = _abundance_series(test)

    mask = (
        np.isfinite(feh_ref)
        & np.isfinite(feh_test)
        & np.isfinite(ofe_ref)
        & np.isfinite(ofe_test)
    )
    d_ofe = np.abs(ofe_test[mask] - ofe_ref[mask])
    d_feh = np.abs(feh_test[mask] - feh_ref[mask])

    return CacheValidationResult(
        n_points=int(mask.sum()),
        ofe_max_abs_diff=float(np.max(d_ofe)),
        ofe_p95_abs_diff=float(np.percentile(d_ofe, 95.0)),
        ofe_mean_abs_diff=float(np.mean(d_ofe)),
        feh_max_abs_diff=float(np.max(d_feh)),
        feh_p95_abs_diff=float(np.percentile(d_feh, 95.0)),
        feh_mean_abs_diff=float(np.mean(d_feh)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate interp cache guard stride against stride=1 reference.")
    ap.add_argument("--stride-test", type=int, required=True, help="Stride value to test (compared to stride=1)")
    ap.add_argument("--endoftime", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--samples", type=int, default=11)
    ap.add_argument("--mass-points", type=int, default=160)
    ap.add_argument("--zeta-points", type=int, default=128)
    ap.add_argument("--binmax-points", type=int, default=128)
    args = ap.parse_args()

    res = validate_cache_stride(
        stride_test=args.stride_test,
        endoftime=args.endoftime,
        tol=args.tol,
        samples=args.samples,
        mass_points=args.mass_points,
        zeta_points=args.zeta_points,
        binmax_points=args.binmax_points,
    )
    print(json.dumps(res.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
