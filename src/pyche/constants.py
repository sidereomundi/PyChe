"""Named constants for element indices and physics values.

These replace magic numbers scattered throughout the codebase.
All element indices use 1-based Fortran indexing.
"""

# ---------------------------------------------------------------------------
# Element indices (1-based, matching Fortran COMMON block layout)
# ---------------------------------------------------------------------------
HEN = 1       # He (net from nucleosynthesis)
C12 = 2       # Carbon-12
O16 = 3       # Oxygen-16
N14 = 4       # Nitrogen-14
C13 = 5       # Carbon-13
NE = 6        # Neon
MG = 7        # Magnesium
SI = 8        # Silicon
FE = 9        # Iron
S14 = 10      # Sulphur-14
C13S = 11     # C13 (secondary)
S32 = 12      # Sulphur-32
CA = 13       # Calcium
REMN = 14     # Remnant mass (special slot, excluded from chemistry sums)
ZN = 15       # Zinc
K = 16        # Potassium
SC = 17       # Scandium
TI = 18       # Titanium
V = 19        # Vanadium
CR = 20       # Chromium
MN = 21       # Manganese
CO = 22       # Cobalt
NI = 23       # Nickel
LA = 24       # Lanthanum
BA = 25       # Barium
EU = 26       # Europium
SR = 27       # Strontium
Y = 28        # Yttrium
ZR = 29       # Zirconium
RB = 30       # Rubidium
LI = 31       # Lithium
H = 32        # Hydrogen  (elem - 1)
HE4_TOTAL = 33  # He-4 total (elem)

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
SUPERF = 20000.0        # Surface normalisation (Msun / pc^2 ?)
SF_THRESHOLD = 0.1      # Star-formation gas threshold
KAPPA = 1.5             # Kennicutt-Schmidt exponent
R_GALACTIC = 8.0        # Galactocentric radius (kpc)
SIGMA_SUN = 50.0        # Solar-neighbourhood surface density

# ---------------------------------------------------------------------------
# Primordial abundances (mass fractions)
# ---------------------------------------------------------------------------
PRIMORDIAL_HE4 = 0.241
PRIMORDIAL_H = 0.759

# ---------------------------------------------------------------------------
# Spallation coefficients  (Li production via cosmic-ray spallation)
# ---------------------------------------------------------------------------
SPALLA_LOG_CONST = -9.50
SPALLA_SLOPE = 1.24
SPALLA_FEH_OFFSET = 2.75

# ---------------------------------------------------------------------------
# Simulation limits / defaults
# ---------------------------------------------------------------------------
GALACTIC_AGE = 13500.0     # Max stellar lifetime threshold (Myr)
GAS_FLOOR = 1.0e-20        # Floor value for gas / metallicity
NMAX_DEFAULT = 15000        # Default time-array size
NUM_ELEMENTS = 33           # Number of tracked element species

# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
INI_LI = 1.0e-9            # Initial Li abundance (ini[31])
