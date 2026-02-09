"""Diagnostic plotting utilities for chemical evolution outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .output_io import FIS_COLUMNS, MOD_COLUMNS
from .output_reader import read_outputs


def _to_numpy(table):
    if hasattr(table, "to_numpy"):
        return table.to_numpy(), list(table.columns)
    return np.asarray(table), None


def _col_idx(columns: list[str], name: str) -> int:
    return columns.index(name)


def create_diagnostic_plots(
    output_dir: str | Path,
    *,
    plot_dir: str | Path | None = None,
    prefer: str = "auto",
    binary_format: str = "pickle",
) -> dict[str, str]:
    """Create standard chemical-evolution diagnostic plots.

    Returns mapping of plot names to file paths.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Plotting requires matplotlib") from exc

    payload = read_outputs(output_dir, prefer=prefer, binary_format=binary_format)
    mod, mod_cols = _to_numpy(payload["mod"])
    fis, fis_cols = _to_numpy(payload["fis"])
    mod_cols = mod_cols or MOD_COLUMNS
    fis_cols = fis_cols or FIS_COLUMNS

    out = Path(plot_dir) if plot_dir is not None else (Path(output_dir) / "plots")
    out.mkdir(parents=True, exist_ok=True)

    time_f = fis[:, _col_idx(fis_cols, "time")]
    sfr = fis[:, _col_idx(fis_cols, "sfr")]
    allv = fis[:, _col_idx(fis_cols, "all")]
    gas = fis[:, _col_idx(fis_cols, "gas")]
    stars = fis[:, _col_idx(fis_cols, "stars")]
    remn = fis[:, _col_idx(fis_cols, "remn")]
    zeta = fis[:, _col_idx(fis_cols, "zeta")]

    time_m = mod[:, _col_idx(mod_cols, "time")]
    fe = mod[:, _col_idx(mod_cols, "Fe")]
    h = mod[:, _col_idx(mod_cols, "H")]
    o16 = mod[:, _col_idx(mod_cols, "O16")]
    mg = mod[:, _col_idx(mod_cols, "Mg")]

    eps = 1.0e-30
    # Model outputs are masses, so use solar mass-ratio anchors for bracket abundances.
    # [Fe/H] = log10((Fe/H)/(Fe/H)_sun), etc.
    FE_H_SUN_MASS = 56.0 * 10.0 ** (-4.50)     # Asplund-like solar Fe/H (number) converted to mass ratio
    O_FE_SUN_MASS = (16.0 / 56.0) * 10.0 ** (8.69 - 7.50)
    MG_FE_SUN_MASS = (24.305 / 56.0) * 10.0 ** (7.60 - 7.50)

    log_feh = np.log10(np.maximum(fe / np.maximum(h, eps), eps) / FE_H_SUN_MASS)
    log_o_fe = np.log10(np.maximum(o16 / np.maximum(fe, eps), eps) / O_FE_SUN_MASS)
    log_mg_fe = np.log10(np.maximum(mg / np.maximum(fe, eps), eps) / MG_FE_SUN_MASS)
    feh_min, feh_max = -5.0, 1.0

    written: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time_f, sfr, lw=1.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("SFR")
    ax.set_title("Star Formation History")
    ax.grid(alpha=0.3)
    p = out / "sfr_vs_time.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written["sfr_vs_time"] = str(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time_f, allv, label="all", lw=1.4)
    ax.plot(time_f, gas, label="gas", lw=1.4)
    ax.plot(time_f, stars, label="stars", lw=1.4)
    ax.plot(time_f, remn, label="remn", lw=1.4)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mass")
    ax.set_title("Mass Budget Evolution")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    p = out / "mass_budget_vs_time.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written["mass_budget_vs_time"] = str(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time_f, zeta, lw=1.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Zeta")
    ax.set_title("Metallicity Evolution")
    ax.grid(alpha=0.3)
    p = out / "metallicity_vs_time.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written["metallicity_vs_time"] = str(p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(time_m, log_feh, lw=1.6)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("[Fe/H]")
    axes[0].set_title("Iron Enrichment")
    axes[0].set_ylim(feh_min, feh_max)
    axes[0].grid(alpha=0.3)

    feh_mask = np.isfinite(log_feh) & (log_feh >= feh_min) & (log_feh <= feh_max)
    axes[1].plot(log_feh[feh_mask], log_o_fe[feh_mask], lw=1.4, label="[O/Fe]")
    axes[1].plot(log_feh[feh_mask], log_mg_fe[feh_mask], lw=1.4, label="[Mg/Fe]")
    axes[1].set_xlabel("[Fe/H]")
    axes[1].set_ylabel("[X/Fe]")
    axes[1].set_title("Abundance Ratio Tracks")
    axes[1].set_xlim(feh_min, feh_max)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3)

    p = out / "abundance_tracks.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written["abundance_tracks"] = str(p)

    # MDF diagnostic: stellar-mass-weighted metallicity distribution over [Fe/H] in [-5, 1].
    stars_delta = np.diff(stars, prepend=stars[0])
    stars_delta = np.maximum(stars_delta, 0.0)
    mdf_mask = np.isfinite(log_feh) & (log_feh >= feh_min) & (log_feh <= feh_max)
    log_feh_sel = log_feh[mdf_mask]
    w_sel = stars_delta[mdf_mask]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(feh_min, feh_max, 61)
    if log_feh_sel.size > 0:
        if np.sum(w_sel) > 0:
            w_sel = w_sel / np.sum(w_sel)
            ax.hist(log_feh_sel, bins=bins, weights=w_sel, histtype="stepfilled", alpha=0.75)
            ax.set_ylabel("Normalized Stellar-Mass Fraction")
        else:
            ax.hist(log_feh_sel, bins=bins, density=True, histtype="stepfilled", alpha=0.75)
            ax.set_ylabel("Density")
    else:
        ax.hist([], bins=bins, histtype="stepfilled", alpha=0.75)
        ax.set_ylabel("Density")
    ax.set_xlabel("[Fe/H]")
    ax.set_title("MDF")
    ax.set_xlim(feh_min, feh_max)
    ax.grid(alpha=0.3)

    p = out / "mdf_feh.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written["mdf_feh"] = str(p)

    return written
