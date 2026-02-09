"""TAU function translated from Fortran.

This module implements the stellar lifetime function `TAU`
as defined in ``src/tau.f90``. The Fortran `COMMON` blocks are
replaced with plain function arguments.
"""

import numpy as np


def tau(mass: float, tautype: int, binmax: float) -> float:
    """Compute stellar lifetime.

    Parameters
    ----------
    mass : float
        Stellar mass.
    tautype : int
        Type selector matching Fortran's ``tautype``.
    binmax : float
        Additional parameter used to adjust the lifetime when dealing with
        binary systems in the original code.

    Returns
    -------
    float
        Lifetime in Myr. The piecewise power-law relations follow directly
        from ``src/tau.f90`` using NumPy for the logarithms.
    """
    if tautype == 1:
        if mass <= 1.3:
            t = 1000 * 10 ** (-0.6545 * np.log10(mass) + 1.0)
        elif mass <= 3.0:
            t = 1000 * 10 ** (-3.7 * np.log10(mass) + 1.35)
        elif mass <= 7.0:
            t = 1000 * 10 ** (-2.51 * np.log10(mass) + 0.77)
        elif mass <= 15.0:
            t = 1000 * 10 ** (-1.78 * np.log10(mass) + 0.17)
        elif mass <= 60.0:
            t = 1000 * 10 ** (-0.86 * np.log10(mass) - 0.94)
        else:
            t = 1000 * (1.2 * mass ** -1.85 + 3e-3)
    else:
        if mass <= 0.56:
            t = 50.0
        elif mass <= 6.6:
            t = 10 ** (
                (
                    0.334
                    - np.sqrt(1.79 - 0.2232 * (7.764 - np.log10(mass)))
                )
                / 0.1116
            )
        else:
            t = 1.2 * mass ** -1.85 + 3e-3
        t *= 1000.0

    if binmax < 0.0:
        if binmax > -8.0:
            t += 500.0
        else:
            t += 4000.0

    return float(t)
