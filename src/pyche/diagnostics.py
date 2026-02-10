"""Diagnostic routines for GCE outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .output_reader import read_outputs


def _to_numpy(table):
    if hasattr(table, "to_numpy"):
        return table.to_numpy()
    return np.asarray(table)


def diagnostics_from_tables(mod, fis) -> dict:
    mod_np = _to_numpy(mod)
    fis_np = _to_numpy(fis)

    final = fis_np[-1]
    checks = {
        "gas_nonnegative": bool(np.all(fis_np[:, 2] >= -1.0e-12)),
        "stars_nonnegative": bool(np.all(fis_np[:, 3] >= -1.0e-12)),
        "remn_nonnegative": bool(np.all(fis_np[:, 4] >= -1.0e-12)),
        "all_nonnegative": bool(np.all(fis_np[:, 1] >= -1.0e-12)),
    }
    return {
        "n_steps": int(fis_np.shape[0]),
        "final_gas": float(final[2]),
        "final_stars": float(final[3]),
        "final_remn": float(final[4]),
        "final_zeta": float(final[6]),
        "max_sfr": float(np.max(fis_np[:, 7])),
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
    }


def run_diagnostics(output_dir: str | Path, *, prefer: str = "auto", binary_format: str = "pickle") -> dict:
    payload = read_outputs(output_dir, prefer=prefer, binary_format=binary_format)
    diag = diagnostics_from_tables(payload["mod"], payload["fis"])
    diag["format"] = payload["format"]
    return diag

