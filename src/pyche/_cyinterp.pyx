# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython interpolation primitives used by cython backend."""

import numpy as np
cimport numpy as cnp


cdef inline double _lin2(double x0, double x1, double y0, double y1, double x):
    cdef double den = x1 - x0
    if den == 0.0:
        return y0
    return y0 + (y1 - y0) * ((x - x0) / den)


cdef inline double _litio(
    cnp.ndarray[cnp.float64_t, ndim=1] massaLi,
    cnp.ndarray[cnp.float64_t, ndim=2] YLi,
    double massa,
):
    cdef int i, indice_m = 1
    for i in range(1, 16):
        if massa >= massaLi[i]:
            indice_m = i
    if indice_m > 14:
        indice_m = 14
    return _lin2(
        massaLi[indice_m],
        massaLi[indice_m + 1],
        YLi[indice_m, 1],
        YLi[indice_m + 1, 1],
        massa,
    )


cdef inline double _bario_component(
    cnp.ndarray[cnp.float64_t, ndim=2] grid,
    int indice_m,
    int indice_z,
    double massa,
    double z0,
    double z1,
    double m0,
    double m1,
    double zcerc,
):
    cdef double q1 = _lin2(z0, z1, grid[indice_m, indice_z], grid[indice_m, indice_z + 1], zcerc)
    cdef double q2 = _lin2(z0, z1, grid[indice_m + 1, indice_z], grid[indice_m + 1, indice_z + 1], zcerc)
    return _lin2(m0, m1, q1, q2, massa)


cdef inline tuple _bario(
    cnp.ndarray[cnp.float64_t, ndim=1] zbario,
    cnp.ndarray[cnp.float64_t, ndim=1] massaba,
    cnp.ndarray[cnp.float64_t, ndim=2] ba,
    cnp.ndarray[cnp.float64_t, ndim=2] sr,
    cnp.ndarray[cnp.float64_t, ndim=2] yt,
    cnp.ndarray[cnp.float64_t, ndim=2] eu,
    cnp.ndarray[cnp.float64_t, ndim=2] zr,
    cnp.ndarray[cnp.float64_t, ndim=2] la,
    cnp.ndarray[cnp.float64_t, ndim=2] rb,
    double zcerc2,
    double massa,
):
    cdef int i, indice_m = 1, indice_z = 1
    cdef double zcerc = zcerc2
    cdef double z0, z1, m0, m1
    cdef double qba, qy, qsr, qeu, qzr, qla, qrb

    for i in range(1, 6):
        if massa >= massaba[i]:
            indice_m = i
    if indice_m > 4:
        indice_m = 4

    if zcerc < zbario[1]:
        zcerc = zbario[1]
    if zcerc > zbario[9]:
        zcerc = zbario[9]

    for i in range(1, 10):
        if zcerc >= zbario[i]:
            indice_z = i
    if indice_z > 8:
        indice_z = 8

    z0 = zbario[indice_z]
    z1 = zbario[indice_z + 1]
    m0 = massaba[indice_m]
    m1 = massaba[indice_m + 1]

    qba = _bario_component(ba, indice_m, indice_z, massa, z0, z1, m0, m1, zcerc)
    qy = _bario_component(yt, indice_m, indice_z, massa, z0, z1, m0, m1, zcerc)
    qsr = _bario_component(sr, indice_m, indice_z, massa, z0, z1, m0, m1, zcerc)
    qeu = _bario_component(eu, indice_m, indice_z, massa, z0, z1, m0, m1, zcerc)
    qzr = _bario_component(zr, indice_m, indice_z, massa, z0, z1, m0, m1, zcerc)
    qla = _bario_component(la, indice_m, indice_z, massa, z0, z1, m0, m1, zcerc)
    qrb = _bario_component(rb, indice_m, indice_z, massa, z0, z1, m0, m1, zcerc)
    return qba, qsr, qy, qeu, qzr, qla, qrb


cpdef tuple polint(cnp.ndarray[cnp.float64_t, ndim=1] xa,
                   cnp.ndarray[cnp.float64_t, ndim=1] ya,
                   double x):
    cdef Py_ssize_t n = xa.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] c = ya.copy()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] d = ya.copy()
    cdef Py_ssize_t i, m, ns = 0
    cdef double y, dy = 0.0, best, diff, ho, hp, w, den

    if n == 0:
        return 0.0, 0.0

    best = abs(x - xa[0])
    for i in range(1, n):
        diff = abs(x - xa[i])
        if diff < best:
            best = diff
            ns = i

    y = ya[ns]
    ns -= 1

    for m in range(1, n):
        for i in range(0, n - m):
            ho = xa[i] - x
            hp = xa[i + m] - x
            w = c[i + 1] - d[i]
            den = ho - hp
            if den == 0.0:
                return y, dy
            den = w / den
            d[i] = hp * den
            c[i] = ho * den
        if 2 * (ns + 1) < (n - m):
            dy = c[ns + 1]
        else:
            dy = d[ns]
            ns -= 1
        y += dy
    return y, dy


cpdef tuple interp_full(
    double mass,
    double zeta,
    double binmax,
    int ninputyield,
    cnp.ndarray[cnp.float64_t, ndim=3] W,
    cnp.ndarray[cnp.float64_t, ndim=1] massa,
    cnp.ndarray[cnp.float64_t, ndim=1] massac,
    cnp.ndarray[cnp.float64_t, ndim=1] MBa,
    cnp.ndarray[cnp.float64_t, ndim=2] WBa,
    cnp.ndarray[cnp.float64_t, ndim=2] WSr,
    cnp.ndarray[cnp.float64_t, ndim=2] WY,
    cnp.ndarray[cnp.float64_t, ndim=2] WLa,
    cnp.ndarray[cnp.float64_t, ndim=2] WZr,
    cnp.ndarray[cnp.float64_t, ndim=2] WRb,
    cnp.ndarray[cnp.float64_t, ndim=2] WEu,
    cnp.ndarray[cnp.float64_t, ndim=1] zbario,
    cnp.ndarray[cnp.float64_t, ndim=1] massaba,
    cnp.ndarray[cnp.float64_t, ndim=2] ba,
    cnp.ndarray[cnp.float64_t, ndim=2] sr,
    cnp.ndarray[cnp.float64_t, ndim=2] yt,
    cnp.ndarray[cnp.float64_t, ndim=2] eu,
    cnp.ndarray[cnp.float64_t, ndim=2] zr,
    cnp.ndarray[cnp.float64_t, ndim=2] la,
    cnp.ndarray[cnp.float64_t, ndim=2] rb,
    cnp.ndarray[cnp.float64_t, ndim=2] YLi,
    cnp.ndarray[cnp.float64_t, ndim=1] massaLi,
):
    cdef int elem = 33, nmax = 23
    cdef double H = mass, ratio = 0.0, hecore = 0.0, value1 = 0.0
    cdef int i, j, k, kk, z, zz, idx
    cdef double met0, met1, mm0, mm1, qbar_i, q1v, q2v, qli
    cdef double qbar = 1.0e-30, qeu2 = 1.0e-30, qla = 1.0e-30, qsrr = 1.0e-30
    cdef double qy = 1.0e-30, qzr = 1.0e-30, qrb = 1.0e-30
    cdef cnp.ndarray[cnp.float64_t, ndim=1] q = np.zeros(elem + 1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] q1 = np.zeros(elem + 1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] q2 = np.zeros(elem + 1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] cosn2 = np.ones(nmax + 1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] qia = np.zeros(elem + 1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] aa = np.zeros(16, dtype=np.float64)
    cdef tuple tmp
    cdef double denom, feh

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

        if H > 8.0:
            aa[1] = 0.0
            aa[2] = 0.02e-4
            aa[3] = 0.02e-2
            aa[4] = 0.02e-1
            aa[5] = 0.02
        else:
            aa[1] = 0.0
            aa[2] = 0.004
            aa[3] = 0.008
            aa[4] = 0.02
            aa[5] = 0.04

        z = 1
        for j in range(1, 6):
            if aa[j] <= zeta:
                z = j
        zz = z if z == 5 else z + 1
        met0 = aa[z]
        met1 = aa[zz]

        k = 1
        for j in range(1, ninputyield + 1):
            if massa[j] < H:
                k = j
        if k < 1:
            k = 1
        if k > ninputyield - 1:
            k = ninputyield - 1
        kk = k + 1
        mm0 = massa[k]
        mm1 = massa[kk]

        for j in range(1, 24):
            q1[j] = _lin2(mm0, mm1, W[j, k, z], W[j, kk, z], H) * cosn2[j]

        if z < 5:
            for j in range(1, 24):
                q2[j] = _lin2(mm0, mm1, W[j, k, zz], W[j, kk, zz], H) * cosn2[j]
            for j in range(1, 24):
                q[j] = _lin2(met0, met1, q1[j], q2[j], zeta)
        else:
            for j in range(1, 24):
                q[j] = q1[j]

        aa[6] = 0.0
        aa[7] = 1.0e-8
        aa[8] = 1.0e-5
        aa[9] = 4.0e-3
        aa[10] = 2.0e-2
        aa[11] = 0.0
        aa[12] = 0.001
        aa[13] = 0.004
        aa[14] = 0.008
        aa[15] = 2.0e-2

        k = 1
        for j in range(1, 14):
            if massac[j] < H:
                k = j
        if k < 1:
            k = 1
        if k > 12:
            k = 12
        kk = k + 1
        mm0 = massac[k]
        mm1 = massac[kk]

        z = 6
        for j in range(6, 11):
            if aa[j] <= zeta:
                z = j
        zz = z + 1
        if zz > 10:
            zz = 10

        for j in range(1, 15):
            q1[j] = _lin2(mm0, mm1, W[j, k, z], W[j, kk, z], H)
        if z < 10:
            for j in range(1, 15):
                q2[j] = _lin2(mm0, mm1, W[j, k, zz], W[j, kk, zz], H)

        aa[1] = 0.0
        aa[2] = 0.001
        aa[3] = 0.004
        aa[4] = 0.02
        aa[5] = 0.05
        z = 1
        for j in range(1, 6):
            if aa[j] <= zeta:
                z = j
        zz = z if z == 5 else z + 1
        met0 = aa[z]
        met1 = aa[zz]

        k = 1
        for j in range(1, ninputyield + 1):
            if massa[j] < H:
                k = j
        if k < 1:
            k = 1
        if k > ninputyield - 1:
            k = ninputyield - 1
        kk = k + 1
        mm0 = massa[k]
        mm1 = massa[kk]

        q1[9] = _lin2(mm0, mm1, W[9, k, z], W[9, kk, z], H)
        q1[21] = _lin2(mm0, mm1, W[21, k, z], W[21, kk, z], H)
        q1[13] = _lin2(mm0, mm1, W[13, k, z], W[13, kk, z], H)
        if z < 5:
            q2[9] = _lin2(mm0, mm1, W[9, k, zz], W[9, kk, zz], H)
            q2[21] = _lin2(mm0, mm1, W[21, k, zz], W[21, kk, zz], H)
            q2[13] = _lin2(mm0, mm1, W[13, k, zz], W[13, kk, zz], H)
            q[9] = _lin2(met0, met1, q1[9], q2[9], zeta)
        else:
            q[9] = q1[9]

        qbar = 1.0e-30
        qeu2 = 1.0e-30
        qla = 1.0e-30
        qsrr = 1.0e-30
        qy = 1.0e-30
        qzr = 1.0e-30
        qrb = 1.0e-30
        if 10.0 <= H <= 30.0:
            value1 = 0.8e-6
            qbar = value1 * 1.0
            qeu2 = value1 * (0.117 * 151.0 / 138.0)
            qla = value1 * 0.136
            qsrr = value1 * (3.16 * 88.0 / 138.0)
            qy = value1 * (1.625 * 89.0 / 138.0 / 3.0)
            qzr = value1 * (2.53 * 90.0 / 138.0)
            qrb = value1 * (3.16 * 86.0 / 138.0)

        q[24] = qla
        q[25] = qbar
        q[26] = qeu2
        q[27] = qsrr
        q[28] = qy
        q[29] = qzr
        q[30] = qrb

        aa[1] = 1.4e-2
        aa[2] = 1.0e-3
        aa[3] = 1.0e-5
        if 15.0 < H < 80.0 and zeta > 1.0e-30:
            k = 4
            for j in range(1, 4):
                if MBa[j] <= H <= MBa[j + 1]:
                    k = j
            if k == 4:
                if H >= 40.0:
                    k = 4
                    kk = 4
                else:
                    k = 1
                    kk = 1
            else:
                kk = k + 1
            mm0 = MBa[k]
            mm1 = MBa[kk]

            if zeta < 1.0e-5:
                q[25] += _lin2(mm0, mm1, WBa[k, 3], WBa[kk, 3], H)
                q[27] += _lin2(mm0, mm1, WSr[k, 3], WSr[kk, 3], H)
                q[28] += _lin2(mm0, mm1, WY[k, 3], WY[kk, 3], H)
                q[24] += _lin2(mm0, mm1, WLa[k, 3], WLa[kk, 3], H)
                q[29] += _lin2(mm0, mm1, WZr[k, 3], WZr[kk, 3], H)
                q[30] += _lin2(mm0, mm1, WRb[k, 3], WRb[kk, 3], H)
                q[26] += _lin2(mm0, mm1, WEu[k, 3], WEu[kk, 3], H)
            elif 1.0e-5 <= zeta < 1.4e-2:
                z = 1
                for j in range(1, 3):
                    if aa[j + 1] < zeta <= aa[j]:
                        z = j
                zz = z + 1
                met0 = aa[zz]
                met1 = aa[z]

                q1v = _lin2(mm0, mm1, WBa[k, zz], WBa[kk, zz], H)
                q2v = _lin2(mm0, mm1, WBa[k, z], WBa[kk, z], H)
                q[25] += _lin2(met0, met1, q1v, q2v, zeta)

                q1v = _lin2(mm0, mm1, WSr[k, zz], WSr[kk, zz], H)
                q2v = _lin2(mm0, mm1, WSr[k, z], WSr[kk, z], H)
                q[27] += _lin2(met0, met1, q1v, q2v, zeta)

                q1v = _lin2(mm0, mm1, WY[k, zz], WY[kk, zz], H)
                q2v = _lin2(mm0, mm1, WY[k, z], WY[kk, z], H)
                q[28] += _lin2(met0, met1, q1v, q2v, zeta)

                q1v = _lin2(mm0, mm1, WLa[k, zz], WLa[kk, zz], H)
                q2v = _lin2(mm0, mm1, WLa[k, z], WLa[kk, z], H)
                q[24] += _lin2(met0, met1, q1v, q2v, zeta)

                q1v = _lin2(mm0, mm1, WZr[k, zz], WZr[kk, zz], H)
                q2v = _lin2(mm0, mm1, WZr[k, z], WZr[kk, z], H)
                q[29] += _lin2(met0, met1, q1v, q2v, zeta)

                q1v = _lin2(mm0, mm1, WRb[k, zz], WRb[kk, zz], H)
                q2v = _lin2(mm0, mm1, WRb[k, z], WRb[kk, z], H)
                q[30] += _lin2(met0, met1, q1v, q2v, zeta)

                q1v = _lin2(mm0, mm1, WEu[k, zz], WEu[kk, zz], H)
                q2v = _lin2(mm0, mm1, WEu[k, z], WEu[kk, z], H)
                q[26] += _lin2(met0, met1, q1v, q2v, zeta)
            elif zeta > 1.4e-2:
                q[25] += _lin2(mm0, mm1, WBa[k, 1], WBa[kk, 1], H)
                q[27] += _lin2(mm0, mm1, WSr[k, 1], WSr[kk, 1], H)
                q[28] += _lin2(mm0, mm1, WY[k, 1], WY[kk, 1], H)
                q[24] += _lin2(mm0, mm1, WLa[k, 1], WLa[kk, 1], H)
                q[29] += _lin2(mm0, mm1, WZr[k, 1], WZr[kk, 1], H)
                q[30] += _lin2(mm0, mm1, WRb[k, 1], WRb[kk, 1], H)
                q[26] += _lin2(mm0, mm1, WEu[k, 1], WEu[kk, 1], H)

        if 1.3 <= H <= 3.0:
            tmp = _bario(zbario, massaba, ba, sr, yt, eu, zr, la, rb, zeta, H)
            qbar = <double>tmp[0]
            qsrr = <double>tmp[1]
            qy = <double>tmp[2]
            qeu2 = <double>tmp[3]
            qzr = <double>tmp[4]
            qla = <double>tmp[5]
            qrb = <double>tmp[6]
        if 1.0 <= H <= 6.0:
            qli = _litio(massaLi, YLi, H)
            if qli > 1.0e-20:
                q[31] = qli
        if 1.3 <= H <= 3.0:
            q[24] = qla / 2.0
            q[25] = qbar / 2.0
            q[26] = qeu2 / 2.0
            q[27] = qsrr / 2.0
            q[28] = qy / 2.0
            q[29] = qzr / 2.0
            q[30] = qrb / 2.0

        if 12.0 <= H <= 50.0:
            q[9] = 0.07
        else:
            q[9] = 1.0e-20

        hecore = 0.0
        for i in range(1, 32):
            if H < 0.5:
                if i == 14:
                    q[i] = H
                else:
                    q[i] = 0.0
            if binmax > 0.0:
                q[i] = qia[i] * ratio + q[i]
            hecore += q[i]
    else:
        q = np.zeros(elem + 1, dtype=np.float64)
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
        hecore = 0.0
        for i in range(1, 32):
            hecore += q[i]

    return q[1:34], hecore


cdef inline int _find_interval(cnp.ndarray[cnp.float64_t, ndim=1] grid, double x):
    cdef int lo = 0
    cdef int hi = <int>grid.shape[0] - 1
    cdef int mid
    if x <= grid[0]:
        return 0
    if x >= grid[hi]:
        return hi - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


cpdef tuple interp_from_cache(
    double mass,
    double zeta,
    double binmax,
    cnp.ndarray[cnp.float64_t, ndim=1] zeta_grid,
    cnp.ndarray[cnp.float64_t, ndim=1] mass_grid,
    cnp.ndarray[cnp.float64_t, ndim=1] binmax_grid,
    cnp.ndarray[cnp.float64_t, ndim=3] cache_zero_q,
    cnp.ndarray[cnp.float64_t, ndim=2] cache_zero_h,
    cnp.ndarray[cnp.float64_t, ndim=3] cache_pos_q_base,
    cnp.ndarray[cnp.float64_t, ndim=2] cache_pos_h_base,
    cnp.ndarray[cnp.float64_t, ndim=1] qia_vec,
    double qia_sum,
):
    cdef int ix, iy, c
    cdef double x0, x1, y0, y1, fx, fy, ratio, h
    cdef cnp.ndarray[cnp.float64_t, ndim=1] q = np.empty(33, dtype=np.float64)
    cdef double v00, v10, v01, v11
    cdef bint used = False

    if zeta < zeta_grid[0] or zeta > zeta_grid[zeta_grid.shape[0] - 1]:
        return q, 0.0, used

    if binmax == 0.0:
        if mass < mass_grid[0] or mass > mass_grid[mass_grid.shape[0] - 1]:
            return q, 0.0, used
        ix = _find_interval(mass_grid, mass)
        iy = _find_interval(zeta_grid, zeta)
        x0 = mass_grid[ix]
        x1 = mass_grid[ix + 1]
        y0 = zeta_grid[iy]
        y1 = zeta_grid[iy + 1]
        fx = 0.0 if x1 == x0 else (mass - x0) / (x1 - x0)
        fy = 0.0 if y1 == y0 else (zeta - y0) / (y1 - y0)
        for c in range(33):
            v00 = cache_zero_q[ix, iy, c]
            v10 = cache_zero_q[ix + 1, iy, c]
            v01 = cache_zero_q[ix, iy + 1, c]
            v11 = cache_zero_q[ix + 1, iy + 1, c]
            q[c] = (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (1.0 - fx) * fy * v01 + fx * fy * v11
        v00 = cache_zero_h[ix, iy]
        v10 = cache_zero_h[ix + 1, iy]
        v01 = cache_zero_h[ix, iy + 1]
        v11 = cache_zero_h[ix + 1, iy + 1]
        h = (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (1.0 - fx) * fy * v01 + fx * fy * v11
        used = True
        return q, h, used

    if binmax > 0.0:
        if binmax < binmax_grid[0] or binmax > binmax_grid[binmax_grid.shape[0] - 1]:
            return q, 0.0, used
        ix = _find_interval(binmax_grid, binmax)
        iy = _find_interval(zeta_grid, zeta)
        x0 = binmax_grid[ix]
        x1 = binmax_grid[ix + 1]
        y0 = zeta_grid[iy]
        y1 = zeta_grid[iy + 1]
        fx = 0.0 if x1 == x0 else (binmax - x0) / (x1 - x0)
        fy = 0.0 if y1 == y0 else (zeta - y0) / (y1 - y0)
        ratio = (mass + binmax) / binmax
        for c in range(33):
            v00 = cache_pos_q_base[ix, iy, c]
            v10 = cache_pos_q_base[ix + 1, iy, c]
            v01 = cache_pos_q_base[ix, iy + 1, c]
            v11 = cache_pos_q_base[ix + 1, iy + 1, c]
            q[c] = (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (1.0 - fx) * fy * v01 + fx * fy * v11
        for c in range(31):
            q[c] += qia_vec[c] * ratio
        v00 = cache_pos_h_base[ix, iy]
        v10 = cache_pos_h_base[ix + 1, iy]
        v01 = cache_pos_h_base[ix, iy + 1]
        v11 = cache_pos_h_base[ix + 1, iy + 1]
        h = (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (1.0 - fx) * fy * v01 + fx * fy * v11
        h += qia_sum * ratio
        used = True
        return q, h, used

    return q, 0.0, used


cpdef double interp_many_full(
    int t,
    double sfr_mass,
    double zeta,
    int elem,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    cnp.ndarray[cnp.float64_t, ndim=1] mstars,
    cnp.ndarray[cnp.float64_t, ndim=1] binmax,
    cnp.ndarray[cnp.float64_t, ndim=1] multi1,
    cnp.ndarray[cnp.float64_t, ndim=1] tdead,
    cnp.ndarray[cnp.float64_t, ndim=1] hecores,
    cnp.ndarray[cnp.float64_t, ndim=1] mstars1_eff,
    cnp.ndarray[cnp.float64_t, ndim=2] qispecial,
    int ninputyield,
    cnp.ndarray[cnp.float64_t, ndim=3] W,
    cnp.ndarray[cnp.float64_t, ndim=1] massa,
    cnp.ndarray[cnp.float64_t, ndim=1] massac,
    cnp.ndarray[cnp.float64_t, ndim=1] MBa,
    cnp.ndarray[cnp.float64_t, ndim=2] WBa,
    cnp.ndarray[cnp.float64_t, ndim=2] WSr,
    cnp.ndarray[cnp.float64_t, ndim=2] WY,
    cnp.ndarray[cnp.float64_t, ndim=2] WLa,
    cnp.ndarray[cnp.float64_t, ndim=2] WZr,
    cnp.ndarray[cnp.float64_t, ndim=2] WRb,
    cnp.ndarray[cnp.float64_t, ndim=2] WEu,
    cnp.ndarray[cnp.float64_t, ndim=1] zbario,
    cnp.ndarray[cnp.float64_t, ndim=1] massaba,
    cnp.ndarray[cnp.float64_t, ndim=2] ba,
    cnp.ndarray[cnp.float64_t, ndim=2] sr,
    cnp.ndarray[cnp.float64_t, ndim=2] yt,
    cnp.ndarray[cnp.float64_t, ndim=2] eu,
    cnp.ndarray[cnp.float64_t, ndim=2] zr,
    cnp.ndarray[cnp.float64_t, ndim=2] la,
    cnp.ndarray[cnp.float64_t, ndim=2] rb,
    cnp.ndarray[cnp.float64_t, ndim=2] YLi,
    cnp.ndarray[cnp.float64_t, ndim=1] massaLi,
):
    cdef Py_ssize_t n, k
    cdef int jj, i
    cdef double bm, mass, he, oldstars_contrib = 0.0
    cdef tuple out
    cdef cnp.ndarray[cnp.float64_t, ndim=1] q

    n = indices.shape[0]
    for k in range(n):
        jj = <int>indices[k]
        if tdead[jj] + t > 13500.0:
            oldstars_contrib += multi1[jj] * sfr_mass

        bm = binmax[jj]
        mass = mstars[jj]
        out = interp_full(
            mass,
            zeta,
            bm,
            ninputyield,
            W,
            massa,
            massac,
            MBa,
            WBa,
            WSr,
            WY,
            WLa,
            WZr,
            WRb,
            WEu,
            zbario,
            massaba,
            ba,
            sr,
            yt,
            eu,
            zr,
            la,
            rb,
            YLi,
            massaLi,
        )
        q = <cnp.ndarray[cnp.float64_t, ndim=1]>out[0]
        he = <double>out[1]

        mstars1_eff[jj] = (bm + mass) if (bm > 0.0) else mass
        hecores[jj] = he
        for i in range(1, elem):
            qispecial[i, jj] = q[i - 1]

    return oldstars_contrib


cpdef double interp_many_cached(
    int t,
    double sfr_mass,
    double zeta,
    int elem,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    cnp.ndarray[cnp.float64_t, ndim=1] mstars,
    cnp.ndarray[cnp.float64_t, ndim=1] binmax,
    cnp.ndarray[cnp.float64_t, ndim=1] multi1,
    cnp.ndarray[cnp.float64_t, ndim=1] tdead,
    cnp.ndarray[cnp.float64_t, ndim=1] hecores,
    cnp.ndarray[cnp.float64_t, ndim=1] mstars1_eff,
    cnp.ndarray[cnp.float64_t, ndim=2] qispecial,
    cnp.ndarray[cnp.float64_t, ndim=1] zeta_grid,
    cnp.ndarray[cnp.float64_t, ndim=1] mass_grid,
    cnp.ndarray[cnp.float64_t, ndim=1] binmax_grid,
    cnp.ndarray[cnp.float64_t, ndim=3] cache_zero_q,
    cnp.ndarray[cnp.float64_t, ndim=2] cache_zero_h,
    cnp.ndarray[cnp.float64_t, ndim=3] cache_pos_q_base,
    cnp.ndarray[cnp.float64_t, ndim=2] cache_pos_h_base,
    cnp.ndarray[cnp.float64_t, ndim=1] qia_vec,
    double qia_sum,
    int ninputyield,
    cnp.ndarray[cnp.float64_t, ndim=3] W,
    cnp.ndarray[cnp.float64_t, ndim=1] massa,
    cnp.ndarray[cnp.float64_t, ndim=1] massac,
    cnp.ndarray[cnp.float64_t, ndim=1] MBa,
    cnp.ndarray[cnp.float64_t, ndim=2] WBa,
    cnp.ndarray[cnp.float64_t, ndim=2] WSr,
    cnp.ndarray[cnp.float64_t, ndim=2] WY,
    cnp.ndarray[cnp.float64_t, ndim=2] WLa,
    cnp.ndarray[cnp.float64_t, ndim=2] WZr,
    cnp.ndarray[cnp.float64_t, ndim=2] WRb,
    cnp.ndarray[cnp.float64_t, ndim=2] WEu,
    cnp.ndarray[cnp.float64_t, ndim=1] zbario,
    cnp.ndarray[cnp.float64_t, ndim=1] massaba,
    cnp.ndarray[cnp.float64_t, ndim=2] ba,
    cnp.ndarray[cnp.float64_t, ndim=2] sr,
    cnp.ndarray[cnp.float64_t, ndim=2] yt,
    cnp.ndarray[cnp.float64_t, ndim=2] eu,
    cnp.ndarray[cnp.float64_t, ndim=2] zr,
    cnp.ndarray[cnp.float64_t, ndim=2] la,
    cnp.ndarray[cnp.float64_t, ndim=2] rb,
    cnp.ndarray[cnp.float64_t, ndim=2] YLi,
    cnp.ndarray[cnp.float64_t, ndim=1] massaLi,
):
    cdef Py_ssize_t n, k
    cdef int jj, i
    cdef double bm, mass, he, oldstars_contrib = 0.0
    cdef tuple out
    cdef cnp.ndarray[cnp.float64_t, ndim=1] q
    cdef bint used

    n = indices.shape[0]
    for k in range(n):
        jj = <int>indices[k]
        if tdead[jj] + t > 13500.0:
            oldstars_contrib += multi1[jj] * sfr_mass

        bm = binmax[jj]
        mass = mstars[jj]
        out = interp_from_cache(
            mass,
            zeta,
            bm,
            zeta_grid,
            mass_grid,
            binmax_grid,
            cache_zero_q,
            cache_zero_h,
            cache_pos_q_base,
            cache_pos_h_base,
            qia_vec,
            qia_sum,
        )
        q = <cnp.ndarray[cnp.float64_t, ndim=1]>out[0]
        he = <double>out[1]
        used = <bint>out[2]
        if not used:
            out = interp_full(
                mass,
                zeta,
                bm,
                ninputyield,
                W,
                massa,
                massac,
                MBa,
                WBa,
                WSr,
                WY,
                WLa,
                WZr,
                WRb,
                WEu,
                zbario,
                massaba,
                ba,
                sr,
                yt,
                eu,
                zr,
                la,
                rb,
                YLi,
                massaLi,
            )
            q = <cnp.ndarray[cnp.float64_t, ndim=1]>out[0]
            he = <double>out[1]

        mstars1_eff[jj] = (bm + mass) if (bm > 0.0) else mass
        hecores[jj] = he
        for i in range(1, elem):
            qispecial[i, jj] = q[i - 1]

    return oldstars_contrib
