# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython engine helpers for MinGCE hot loops."""

import numpy as np
cimport numpy as cnp
from libc.math cimport log10, pow


cpdef int death_enrichment_step(
    int t,
    int endoftime,
    int ss2,
    int elem,
    double gas_t,
    double sfr_mass,
    double norm,
    cnp.ndarray[cnp.float64_t, ndim=1] tdead,
    cnp.ndarray[cnp.float64_t, ndim=1] multi1,
    cnp.ndarray[cnp.float64_t, ndim=1] binmax,
    cnp.ndarray[cnp.float64_t, ndim=1] mstars1_eff,
    cnp.ndarray[cnp.float64_t, ndim=1] hecores,
    cnp.ndarray[cnp.float64_t, ndim=2] qispecial,
    cnp.ndarray[cnp.float64_t, ndim=1] qacc,
    cnp.ndarray[cnp.float64_t, ndim=2] qqn,
    cnp.ndarray[cnp.float64_t, ndim=1] stars,
    cnp.ndarray[cnp.float64_t, ndim=1] remn,
    cnp.ndarray[cnp.float64_t, ndim=1] snianum,
):
    cdef int t3 = t
    cdef int jj = 1
    cdef int i
    cdef double next_dead, dm, q14
    cdef bint died_now, next_died
    cdef double starstot = sfr_mass * norm
    cdef double difftot = sfr_mass * norm
    cdef double snian = 0.0
    cdef double gas_inv = 0.0
    if gas_t > 0.0:
        gas_inv = 1.0 / gas_t
    for i in range(elem + 1):
        qacc[i] = 0.0

    while True:
        next_dead = tdead[jj + 1] if (jj + 1) <= ss2 else 1.0e30
        died_now = t3 >= (t + tdead[jj])
        next_died = t3 >= (t + next_dead)

        if died_now:
            dm = multi1[jj] * sfr_mass
            for i in range(1, elem):
                qacc[i] += qispecial[i, jj] * dm
            difftot -= (mstars1_eff[jj] - hecores[jj]) * dm
            starstot -= mstars1_eff[jj] * dm
            if binmax[jj] > 0.0:
                snian += dm

        if (not next_died) and gas_inv > 0.0:
            for i in range(1, 14):
                qqn[i, t3] = qqn[i, t3] + qacc[i] - qqn[i, t] * difftot * gas_inv
            for i in range(15, elem):
                qqn[i, t3] = qqn[i, t3] + qacc[i] - qqn[i, t] * difftot * gas_inv
            qqn[31, t3] = qqn[31, t3] - qqn[31, t] * (mstars1_eff[jj] + hecores[jj]) * multi1[jj] * sfr_mass * gas_inv

        if died_now and next_died:
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

        if t3 >= endoftime:
            break

        t3 += 1
        if died_now:
            jj += 1
            if jj > ss2:
                break

    return 0


cdef inline double _spalla_eval(
    double q9,
    double denom,
    bint use_lut,
    cnp.ndarray[cnp.float64_t, ndim=2] lut,
    double logq_min,
    double logq_max,
    double logd_min,
    double logd_max,
):
    cdef int nq, nd, iq, id
    cdef double lq, ld, fq, fd, v00, v10, v01, v11
    q9 = q9 if q9 > 1.0e-30 else 1.0e-30
    denom = denom if denom > 1.0e-30 else 1.0e-30
    if use_lut:
        nq = <int>lut.shape[0]
        nd = <int>lut.shape[1]
        if nq >= 2 and nd >= 2:
            lq = log10(q9)
            ld = log10(denom)
            if lq >= logq_min and lq <= logq_max and ld >= logd_min and ld <= logd_max:
                fq = (lq - logq_min) * (nq - 1) / (logq_max - logq_min)
                fd = (ld - logd_min) * (nd - 1) / (logd_max - logd_min)
                iq = <int>fq
                id = <int>fd
                if iq >= nq - 1:
                    iq = nq - 2
                    fq = 1.0
                else:
                    fq = fq - iq
                if id >= nd - 1:
                    id = nd - 2
                    fd = 1.0
                else:
                    fd = fd - id
                v00 = lut[iq, id]
                v10 = lut[iq + 1, id]
                v01 = lut[iq, id + 1]
                v11 = lut[iq + 1, id + 1]
                return (1.0 - fq) * (1.0 - fd) * v00 + fq * (1.0 - fd) * v10 + (1.0 - fq) * fd * v01 + fq * fd * v11
    return pow(10.0, -9.50 + 1.24 * (log10(q9 / denom) - (-2.75)) + log10(denom))


cpdef int wind_chem_step_no_wind(
    int t,
    int endoftime,
    int elem,
    int dt_scale,
    double sfr,
    double gas_floor,
    double windist,
    double wind_scale,
    cnp.ndarray[cnp.float64_t, ndim=1] gas,
    cnp.ndarray[cnp.float64_t, ndim=1] allv,
    cnp.ndarray[cnp.float64_t, ndim=1] ini,
    cnp.ndarray[cnp.float64_t, ndim=2] qqn,
    cnp.ndarray[cnp.float64_t, ndim=1] wind,
    cnp.ndarray[cnp.float64_t, ndim=1] zeta,
    cnp.ndarray[cnp.float64_t, ndim=1] spalla,
    int spalla_stride,
    double spalla_inactive_threshold,
    bint use_spalla_lut,
    cnp.ndarray[cnp.float64_t, ndim=2] spalla_lut,
    double spalla_lut_logq_min,
    double spalla_lut_logq_max,
    double spalla_lut_logd_min,
    double spalla_lut_logd_max,
):
    cdef int t3 = t
    cdef int i
    cdef double zsum, denom, spv
    cdef int t_prev = t - dt_scale
    cdef double all_delta = 0.0
    cdef double li_infall
    cdef bint do_sfr = sfr > 0.0
    cdef bint use_spalla = t > 1
    cdef bint do_spalla
    if t_prev < 0:
        t_prev = 0
    if use_spalla:
        all_delta = allv[t] - allv[t_prev]
        li_infall = all_delta * ini[31]
    else:
        li_infall = allv[t] * ini[31]

    if use_spalla:
        while True:
            if do_sfr and gas[t3] > 0.0:
                zsum = 0.0
                for i in range(2, 14):
                    zsum += qqn[i, t3]
                for i in range(15, elem - 1):
                    zsum += qqn[i, t3]
                zeta[t3] = zsum / gas[t3]
            else:
                zeta[t3] = gas_floor

            qqn[elem, t3] = gas[t3] * 0.241 + qqn[1, t3]
            qqn[elem - 1, t3] = gas[t3] * (0.759 - zeta[t3]) - qqn[1, t3]

            qqn[31, t3] = qqn[31, t3] + li_infall
            denom = qqn[elem - 1, t3]
            if denom < 1.0e-30:
                denom = 1.0e-30
            do_spalla = ((t3 - t) % spalla_stride == 0)
            if spalla_inactive_threshold > 0.0 and qqn[31, t3] < spalla_inactive_threshold and qqn[9, t3] < spalla_inactive_threshold:
                do_spalla = False
            if do_spalla:
                spv = _spalla_eval(
                    qqn[9, t3],
                    denom,
                    use_spalla_lut,
                    spalla_lut,
                    spalla_lut_logq_min,
                    spalla_lut_logq_max,
                    spalla_lut_logd_min,
                    spalla_lut_logd_max,
                )
                spalla[t3] = spv
            else:
                spalla[t3] = spalla[t3 - 1]
            qqn[31, t3] = qqn[31, t3] + spalla[t3] - spalla[t3 - 1]

            if t3 >= endoftime:
                break
            t3 += 1
    else:
        while True:
            if do_sfr and gas[t3] > 0.0:
                zsum = 0.0
                for i in range(2, 14):
                    zsum += qqn[i, t3]
                for i in range(15, elem - 1):
                    zsum += qqn[i, t3]
                zeta[t3] = zsum / gas[t3]
            else:
                zeta[t3] = gas_floor

            qqn[elem, t3] = gas[t3] * 0.241 + qqn[1, t3]
            qqn[elem - 1, t3] = gas[t3] * (0.759 - zeta[t3]) - qqn[1, t3]
            qqn[31, t3] = qqn[31, t3] + li_infall

            if t3 >= endoftime:
                break
            t3 += 1

    return 0


cpdef int wind_chem_step_with_wind(
    int t,
    int endoftime,
    int elem,
    int dt_scale,
    double sfr,
    double gas_floor,
    double windist,
    double wind_scale,
    cnp.ndarray[cnp.float64_t, ndim=1] gas,
    cnp.ndarray[cnp.float64_t, ndim=1] allv,
    cnp.ndarray[cnp.float64_t, ndim=1] ini,
    cnp.ndarray[cnp.float64_t, ndim=1] winds,
    cnp.ndarray[cnp.float64_t, ndim=2] qqn,
    cnp.ndarray[cnp.float64_t, ndim=1] wind,
    cnp.ndarray[cnp.float64_t, ndim=1] zeta,
    cnp.ndarray[cnp.float64_t, ndim=1] spalla,
    int spalla_stride,
    double spalla_inactive_threshold,
    bint use_spalla_lut,
    cnp.ndarray[cnp.float64_t, ndim=2] spalla_lut,
    double spalla_lut_logq_min,
    double spalla_lut_logq_max,
    double spalla_lut_logd_min,
    double spalla_lut_logd_max,
):
    cdef int t3 = t
    cdef int i
    cdef double zsum, denom, spv
    cdef int t_prev = t - dt_scale
    cdef double all_delta = 0.0
    cdef double li_infall
    cdef cnp.ndarray[cnp.float64_t, ndim=1] wind_delta = np.zeros(32, dtype=np.float64)
    cdef bint do_sfr = sfr > 0.0
    cdef bint use_spalla = t > 1
    cdef bint do_spalla
    if t_prev < 0:
        t_prev = 0
    if use_spalla:
        all_delta = allv[t] - allv[t_prev]
        li_infall = all_delta * ini[31]
    else:
        li_infall = allv[t] * ini[31]

    for i in range(1, 14):
        wind_delta[i] = qqn[i, t] * winds[i] * wind_scale
    for i in range(15, 32):
        wind_delta[i] = qqn[i, t] * winds[i] * wind_scale

    if use_spalla:
        while True:
            wind[t3] += windist
            for i in range(1, 14):
                qqn[i, t3] = qqn[i, t3] - wind_delta[i]
                if qqn[i, t3] <= gas_floor:
                    qqn[i, t3] = gas_floor
            for i in range(15, 32):
                qqn[i, t3] = qqn[i, t3] - wind_delta[i]
                if qqn[i, t3] <= gas_floor:
                    qqn[i, t3] = gas_floor

            if do_sfr and gas[t3] > 0.0:
                zsum = 0.0
                for i in range(2, 14):
                    zsum += qqn[i, t3]
                for i in range(15, elem - 1):
                    zsum += qqn[i, t3]
                zeta[t3] = zsum / gas[t3]
            else:
                zeta[t3] = gas_floor

            qqn[elem, t3] = gas[t3] * 0.241 + qqn[1, t3]
            qqn[elem - 1, t3] = gas[t3] * (0.759 - zeta[t3]) - qqn[1, t3]

            qqn[31, t3] = qqn[31, t3] + li_infall
            denom = qqn[elem - 1, t3]
            if denom < 1.0e-30:
                denom = 1.0e-30
            do_spalla = ((t3 - t) % spalla_stride == 0)
            if spalla_inactive_threshold > 0.0 and qqn[31, t3] < spalla_inactive_threshold and qqn[9, t3] < spalla_inactive_threshold:
                do_spalla = False
            if do_spalla:
                spv = _spalla_eval(
                    qqn[9, t3],
                    denom,
                    use_spalla_lut,
                    spalla_lut,
                    spalla_lut_logq_min,
                    spalla_lut_logq_max,
                    spalla_lut_logd_min,
                    spalla_lut_logd_max,
                )
                spalla[t3] = spv
            else:
                spalla[t3] = spalla[t3 - 1]
            qqn[31, t3] = qqn[31, t3] + spalla[t3] - spalla[t3 - 1]

            if t3 >= endoftime:
                break
            t3 += 1
    else:
        while True:
            wind[t3] += windist
            for i in range(1, 14):
                qqn[i, t3] = qqn[i, t3] - wind_delta[i]
                if qqn[i, t3] <= gas_floor:
                    qqn[i, t3] = gas_floor
            for i in range(15, 32):
                qqn[i, t3] = qqn[i, t3] - wind_delta[i]
                if qqn[i, t3] <= gas_floor:
                    qqn[i, t3] = gas_floor

            if do_sfr and gas[t3] > 0.0:
                zsum = 0.0
                for i in range(2, 14):
                    zsum += qqn[i, t3]
                for i in range(15, elem - 1):
                    zsum += qqn[i, t3]
                zeta[t3] = zsum / gas[t3]
            else:
                zeta[t3] = gas_floor

            qqn[elem, t3] = gas[t3] * 0.241 + qqn[1, t3]
            qqn[elem - 1, t3] = gas[t3] * (0.759 - zeta[t3]) - qqn[1, t3]
            qqn[31, t3] = qqn[31, t3] + li_infall

            if t3 >= endoftime:
                break
            t3 += 1

    return 0


cpdef int wind_chem_step(
    int t,
    int endoftime,
    int elem,
    int dt_scale,
    double sfr,
    double gas_floor,
    double windist,
    double wind_scale,
    cnp.ndarray[cnp.float64_t, ndim=1] gas,
    cnp.ndarray[cnp.float64_t, ndim=1] allv,
    cnp.ndarray[cnp.float64_t, ndim=1] ini,
    cnp.ndarray[cnp.float64_t, ndim=1] winds,
    cnp.ndarray[cnp.float64_t, ndim=2] qqn,
    cnp.ndarray[cnp.float64_t, ndim=1] wind,
    cnp.ndarray[cnp.float64_t, ndim=1] zeta,
    cnp.ndarray[cnp.float64_t, ndim=1] spalla,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] empty_lut = np.empty((0, 0), dtype=np.float64)
    if wind_scale == 0.0:
        return wind_chem_step_no_wind(
            t, endoftime, elem, dt_scale, sfr, gas_floor, windist, wind_scale,
            gas, allv, ini, qqn, wind, zeta, spalla,
            1, 0.0, False, empty_lut, -30.0, 2.0, -30.0, 2.0
        )
    return wind_chem_step_with_wind(
        t, endoftime, elem, dt_scale, sfr, gas_floor, windist, wind_scale,
        gas, allv, ini, winds, qqn, wind, zeta, spalla,
        1, 0.0, False, empty_lut, -30.0, 2.0, -30.0, 2.0
    )
