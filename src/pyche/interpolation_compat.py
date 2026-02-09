"""Fortran-equation-compatible interpolation kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .model_tables import ModelTables


@dataclass
class CompatInterpolator:
    """Exact-logic interpolation path operating on immutable ``ModelTables``."""

    tables: ModelTables

    @staticmethod
    def _lin2(x0: float, x1: float, y0: float, y1: float, x: float) -> float:
        den = x1 - x0
        if den == 0.0:
            return float(y0)
        return float(y0 + (y1 - y0) * ((x - x0) / den))

    def polint(self, xa: np.ndarray, ya: np.ndarray, x: float) -> Tuple[float, float]:
        n = len(xa)
        c = ya.astype(float).copy()
        d = ya.astype(float).copy()
        ns = int(np.argmin(np.abs(x - xa)))
        y = float(ya[ns])
        ns -= 1
        dy = 0.0
        for m in range(1, n):
            for i in range(n - m):
                ho = xa[i] - x
                hp = xa[i + m] - x
                w = c[i + 1] - d[i]
                den = ho - hp
                if den == 0.0:
                    return y, dy
                den = w / den
                d[i] = hp * den
                c[i] = ho * den
            if 2 * (ns + 1) < n - m:
                dy = float(c[ns + 1])
            else:
                dy = float(d[ns])
                ns -= 1
            y += dy
        return float(y), float(dy)

    def litio(self, zcerc2: float, massa: float) -> float:
        s = self.tables
        indice_m = 1
        for i in range(1, 16):
            if massa >= s.massaLi[i]:
                indice_m = i
        indice_m = min(indice_m, 14)

        return self._lin2(
            float(s.massaLi[indice_m]),
            float(s.massaLi[indice_m + 1]),
            float(s.YLi[indice_m, 1]),
            float(s.YLi[indice_m + 1, 1]),
            float(massa),
        )

    def _bario_component(
        self,
        grid: np.ndarray,
        indice_m: int,
        indice_z: int,
        massa: float,
        z0: float,
        z1: float,
        m0: float,
        m1: float,
        zcerc: float,
    ) -> float:
        q1 = self._lin2(
            z0,
            z1,
            float(grid[indice_m, indice_z]),
            float(grid[indice_m, indice_z + 1]),
            zcerc,
        )
        q2 = self._lin2(
            z0,
            z1,
            float(grid[indice_m + 1, indice_z]),
            float(grid[indice_m + 1, indice_z + 1]),
            zcerc,
        )
        return self._lin2(m0, m1, q1, q2, massa)

    def bario(self, zcerc2: float, massa: float) -> Tuple[float, float, float, float, float, float, float]:
        s = self.tables
        zcerc = float(zcerc2)

        indice_m = 1
        indice_z = 1
        for i in range(1, 6):
            if massa >= s.massaba[i]:
                indice_m = i
        indice_m = min(indice_m, 4)

        if zcerc < s.zbario[1]:
            zcerc = s.zbario[1]
        if zcerc > s.zbario[9]:
            zcerc = s.zbario[9]

        for i in range(1, 10):
            if zcerc >= s.zbario[i]:
                indice_z = i
        indice_z = min(indice_z, 8)

        z0 = float(s.zbario[indice_z])
        z1 = float(s.zbario[indice_z + 1])
        m0 = float(s.massaba[indice_m])
        m1 = float(s.massaba[indice_m + 1])

        qba = self._bario_component(s.ba, indice_m, indice_z, massa, z0, z1, m0, m1, zcerc)
        qy = self._bario_component(s.yt, indice_m, indice_z, massa, z0, z1, m0, m1, zcerc)
        qsr = self._bario_component(s.sr, indice_m, indice_z, massa, z0, z1, m0, m1, zcerc)
        qeu = self._bario_component(s.eu, indice_m, indice_z, massa, z0, z1, m0, m1, zcerc)
        qzr = self._bario_component(s.zr, indice_m, indice_z, massa, z0, z1, m0, m1, zcerc)
        qla = self._bario_component(s.la, indice_m, indice_z, massa, z0, z1, m0, m1, zcerc)
        qrb = self._bario_component(s.rb, indice_m, indice_z, massa, z0, z1, m0, m1, zcerc)
        return qba, qsr, qy, qeu, qzr, qla, qrb

    def interp(self, mass: float, zeta: float, binmax: float) -> Tuple[np.ndarray, float]:
        s = self.tables
        elem = 33
        nmax = 23

        H = float(mass)
        q = np.zeros(elem + 1, dtype=float)
        q1 = np.zeros(elem + 1, dtype=float)
        q2 = np.zeros(elem + 1, dtype=float)
        cosn2 = np.ones(nmax + 1, dtype=float)

        qia = np.zeros(elem + 1, dtype=float)
        qia[1] = 0.0
        qia[2] = 0.048
        qia[3] = 0.143
        qia[4] = 1.16e-6
        qia[5] = 1.40e-6
        qia[6] = 0.00202
        qia[7] = 0.0425 * 1.2
        qia[8] = 0.154
        qia[9] = 0.6
        qia[10] = 0.0
        qia[11] = 0.0
        qia[12] = 0.0846
        qia[13] = 0.0119
        qia[14] = 0.0
        qia[15] = 6.345e-4
        qia[16] = 1.19e-3
        qia[17] = 1.28e-5
        qia[18] = 4.16e-4
        qia[19] = 7.49e-5
        qia[20] = 5.67e-3
        qia[21] = 6.21e-3
        qia[22] = 1.56e-3
        qia[23] = 1.83e-2

        cosn2[3] = 1.0
        cosn2[7] = 6.0
        cosn2[8] = 1.0
        cosn2[13] = 1.0
        cosn2[23] = 1.0
        cosn2[16] = 2.0
        cosn2[17] = 1.15
        cosn2[18] = 15.0
        cosn2[20] = 3.0
        cosn2[22] = 0.3
        cosn2[19] = 4.5

        if binmax >= 0.0:
            if binmax > 0.0:
                ratio = (H + binmax) / binmax
                H = binmax
            else:
                ratio = 0.0

            aa = np.zeros(16, dtype=float)
            aa[1:6] = [0.0, 0.02e-4, 0.02e-2, 0.02e-1, 0.02] if H > 8.0 else [0.0, 0.004, 0.008, 0.02, 0.04]

            z = 1
            for j in range(1, 6):
                if aa[j] <= zeta:
                    z = j
            zz = z if z == 5 else z + 1
            met0 = float(aa[z])
            met1 = float(aa[zz])

            k = 1
            for j in range(1, s.ninputyield + 1):
                if s.massa[j] < H:
                    k = j
            k = max(1, min(k, s.ninputyield - 1))
            kk = k + 1
            mm0 = float(s.massa[k])
            mm1 = float(s.massa[kk])

            for j in range(1, 24):
                q1[j] = self._lin2(mm0, mm1, float(s.W[j, k, z]), float(s.W[j, kk, z]), H) * cosn2[j]

            if z < 5:
                for j in range(1, 24):
                    q2[j] = self._lin2(mm0, mm1, float(s.W[j, k, zz]), float(s.W[j, kk, zz]), H) * cosn2[j]
                for j in range(1, 24):
                    q[j] = self._lin2(met0, met1, float(q1[j]), float(q2[j]), zeta)
            else:
                q[1:24] = q1[1:24]

            aa[6:11] = [0.0, 1.0e-8, 1.0e-5, 4.0e-3, 2.0e-2]
            aa[11:16] = [0.0, 0.001, 0.004, 0.008, 2.0e-2]

            k = 1
            for j in range(1, 14):
                if s.massac[j] < H:
                    k = j
            k = max(1, min(k, 12))
            kk = k + 1
            mm0 = float(s.massac[k])
            mm1 = float(s.massac[kk])

            z = 6
            for j in range(6, 11):
                if aa[j] <= zeta:
                    z = j
            zz = min(z + 1, 10)

            for j in range(1, 15):
                q1[j] = self._lin2(mm0, mm1, float(s.W[j, k, z]), float(s.W[j, kk, z]), H)
            if z < 10:
                for j in range(1, 15):
                    q2[j] = self._lin2(mm0, mm1, float(s.W[j, k, zz]), float(s.W[j, kk, zz]), H)

            aa[1:6] = [0.0, 0.001, 0.004, 0.02, 0.05]
            z = 1
            for j in range(1, 6):
                if aa[j] <= zeta:
                    z = j
            zz = z if z == 5 else z + 1
            met0 = float(aa[z])
            met1 = float(aa[zz])

            k = 1
            for j in range(1, s.ninputyield + 1):
                if s.massa[j] < H:
                    k = j
            k = max(1, min(k, s.ninputyield - 1))
            kk = k + 1
            mm0 = float(s.massa[k])
            mm1 = float(s.massa[kk])

            for idx in (9, 21, 13):
                q1[idx] = self._lin2(mm0, mm1, float(s.W[idx, k, z]), float(s.W[idx, kk, z]), H)
            if z < 5:
                for idx in (9, 21, 13):
                    q2[idx] = self._lin2(mm0, mm1, float(s.W[idx, k, zz]), float(s.W[idx, kk, zz]), H)
                q[9] = self._lin2(met0, met1, float(q1[9]), float(q2[9]), zeta)
            else:
                q[9] = q1[9]

            qbar = qeu2 = qla = qsrr = qy = qzr = qrb = 1.0e-30
            if 10.0 <= H <= 30.0:
                value1 = 0.8e-6
                qbar = value1 * 1.0
                qeu2 = value1 * (0.117 * 151.0 / 138.0)
                qla = value1 * 0.136
                qsrr = value1 * (3.16 * 88.0 / 138.0)
                qy = value1 * (1.625 * 89.0 / 138.0 / 3.0)
                qzr = value1 * (2.53 * 90.0 / 138.0)
                qrb = value1 * (3.16 * 86.0 / 138.0)

            q[24:31] = [qla, qbar, qeu2, qsrr, qy, qzr, qrb]

            aa[1:4] = [1.4e-2, 1.0e-3, 1.0e-5]
            if 15.0 < H < 80.0 and zeta > 1.0e-30:
                k = 4
                for j in range(1, 4):
                    if s.MBa[j] <= H <= s.MBa[j + 1]:
                        k = j
                if k == 4:
                    if H >= 40.0:
                        k = kk = 4
                    else:
                        k = kk = 1
                else:
                    kk = k + 1
                mm0 = float(s.MBa[k])
                mm1 = float(s.MBa[kk])

                grids = ((25, s.WBa), (27, s.WSr), (28, s.WY), (24, s.WLa), (29, s.WZr), (30, s.WRb), (26, s.WEu))
                if zeta < 1.0e-5:
                    for elem_idx, grid in grids:
                        qbar_i = self._lin2(mm0, mm1, float(grid[k, 3]), float(grid[kk, 3]), H)
                        q[elem_idx] += qbar_i
                elif 1.0e-5 <= zeta < 1.4e-2:
                    z = 1
                    for j in range(1, 3):
                        if aa[j + 1] < zeta <= aa[j]:
                            z = j
                    zz = z + 1
                    met0 = float(aa[zz])
                    met1 = float(aa[z])
                    for elem_idx, grid in grids:
                        q1v = self._lin2(mm0, mm1, float(grid[k, zz]), float(grid[kk, zz]), H)
                        q2v = self._lin2(mm0, mm1, float(grid[k, z]), float(grid[kk, z]), H)
                        qbar_i = self._lin2(met0, met1, q1v, q2v, zeta)
                        q[elem_idx] += qbar_i
                elif zeta > 1.4e-2:
                    for elem_idx, grid in grids:
                        qbar_i = self._lin2(mm0, mm1, float(grid[k, 1]), float(grid[kk, 1]), H)
                        q[elem_idx] += qbar_i

            if 1.3 <= H <= 3.0:
                qbar, qsrr, qy, qeu2, qzr, qla, qrb = self.bario(zeta, H)
            if 1.0 <= H <= 6.0:
                qli = self.litio(zeta, H)
                if qli > 1.0e-20:
                    q[31] = qli
            if 1.3 <= H <= 3.0:
                q[24:31] = [qla / 2.0, qbar / 2.0, qeu2 / 2.0, qsrr / 2.0, qy / 2.0, qzr / 2.0, qrb / 2.0]

            q[9] = 0.07 if (12.0 <= H <= 50.0) else 1.0e-20

            hecore = 0.0
            for i in range(1, 32):
                if H < 0.5:
                    q[i] = H if i == 14 else 0.0
                if binmax > 0.0:
                    q[i] = qia[i] * ratio + q[i]
                hecore += q[i]
        else:
            q = np.zeros(elem + 1, dtype=float)
            if binmax >= -8.0:
                q[31] = 2.0e-6 * 4.0
            else:
                value1 = 0.8e-6 * 20.0
                q[24] = value1 * 0.136
                q[25] = value1
                q[26] = value1 * (0.117 * 151.0 / 138.0)
                q[27] = value1 * (3.16 * 88.0 / 138.0)
                q[28] = value1 * (1.625 * 89.0 / 138.0 / 3.0)
                q[29] = value1 * (2.53 * 90.0 / 138.0)
                q[30] = value1 * (3.16 * 86.0 / 138.0)
            hecore = float(np.sum(q[1:32]))

        return q[1:34].copy(), float(hecore)
