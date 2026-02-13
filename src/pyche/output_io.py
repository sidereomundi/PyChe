"""Output writing helpers for legacy and dataframe formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np

MOD_COLUMNS = [
    "time",
    "all",
    "gas",
    "stars",
    "sfr",
    "oldstars",
    "HeN",
    "C12",
    "O16",
    "N14",
    "C13",
    "Ne",
    "Mg",
    "Si",
    "Fe",
    "S14",
    "C13S",
    "S32",
    "Ca",
    "Remn",
    "Zn",
    "K",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Co",
    "Ni",
    "La",
    "Ba",
    "Eu",
    "Sr",
    "Y",
    "Zr",
    "Rb",
    "Li",
    "H",
    "He4",
]

FIS_COLUMNS = [
    "time",
    "all",
    "gas",
    "stars",
    "remn",
    "hot",
    "zeta",
    "sfr",
    "nume",
    "sfr2",
    "snia_num",
    "snia_rate",
]


def build_mod_rows(endoftime: int, allv: np.ndarray, gas: np.ndarray, stars: np.ndarray, sfr_hist: np.ndarray, oldstars: np.ndarray, qqn: np.ndarray) -> np.ndarray:
    rows = np.zeros((endoftime, len(MOD_COLUMNS)), dtype=float)
    s = slice(1, endoftime + 1)
    rows[:, 0] = np.arange(1, endoftime + 1, dtype=float)
    rows[:, 1] = allv[s]
    rows[:, 2] = gas[s]
    rows[:, 3] = stars[s]
    rows[:, 4] = sfr_hist[s]
    rows[:, 5] = oldstars[s]
    rows[:, 6:] = qqn[1:34, s].T
    return rows


def build_fis_rows(endoftime: int, allv: np.ndarray, gas: np.ndarray, stars: np.ndarray, remn: np.ndarray, hot: np.ndarray, zeta: np.ndarray, sfr_hist: np.ndarray, snianum: np.ndarray) -> np.ndarray:
    rows = np.zeros((endoftime, len(FIS_COLUMNS)), dtype=float)
    s = slice(1, endoftime + 1)
    rows[:, 0] = np.arange(1, endoftime + 1, dtype=float)
    rows[:, 1] = allv[s]
    rows[:, 2] = gas[s]
    rows[:, 3] = stars[s]
    rows[:, 4] = remn[s]
    rows[:, 5] = hot[s]
    rows[:, 6] = zeta[s]
    rows[:, 7] = sfr_hist[s]
    rows[:, 8] = 1.0
    rows[:, 9] = 20.0 * (stars[s] - stars[0:endoftime])
    rows[:, 10] = snianum[s]
    rows[:, 11] = snianum[s] - snianum[0:endoftime]
    return rows


def _write_legacy(out_dir: Path, mod_rows: np.ndarray, fis_rows: np.ndarray) -> None:
    with open(out_dir / "fis.encesmin.dat", "w", encoding="ascii") as f_fis:
        f_fis.write("time all gas stars remn hot zeta SFR nume SFR2 SIaN SIar\n")
        for row in fis_rows:
            f_fis.write(" ".join(f"{v: .5e}" for v in row) + "\n")

    with open(out_dir / "modencesmin.dat", "w", encoding="ascii") as f_mod:
        f_mod.write("time all gas star SFR run Hen C12 O16 N14 C13 Ne Mg Si Fe S14 C13S S32 Ca Remn Zn K Sc Ti V Cr Mn Co Ni La Ba Eu Sr Y Zr Rb Li H He4\n")
        for row in mod_rows:
            f_mod.write(" ".join(f"{v: .5e}" for v in row) + "\n")


def _write_dataframe(out_dir: Path, mod_rows: np.ndarray, fis_rows: np.ndarray, *, binary_format: str, write_csv: bool) -> None:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Dataframe output requires pandas installed") from exc

    mod_df = pd.DataFrame(mod_rows, columns=MOD_COLUMNS)
    fis_df = pd.DataFrame(fis_rows, columns=FIS_COLUMNS)

    if binary_format == "pickle":
        mod_df.to_pickle(out_dir / "modencesmin_df.pkl")
        fis_df.to_pickle(out_dir / "fis.encesmin_df.pkl")
    elif binary_format == "parquet":
        mod_df.to_parquet(out_dir / "modencesmin_df.parquet", index=False)
        fis_df.to_parquet(out_dir / "fis.encesmin_df.parquet", index=False)

    if write_csv:
        mod_df.to_csv(out_dir / "modencesmin_df.csv", index=False)
        fis_df.to_csv(out_dir / "fis.encesmin_df.csv", index=False)


def write_outputs(
    out_dir: Path,
    *,
    output_mode: str,
    df_binary_format: str,
    df_write_csv: bool,
    mod_rows: np.ndarray,
    fis_rows: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_mode in {"legacy", "both"}:
        _write_legacy(out_dir, mod_rows, fis_rows)
    if output_mode in {"dataframe", "both"}:
        _write_dataframe(out_dir, mod_rows, fis_rows, binary_format=df_binary_format, write_csv=df_write_csv)

