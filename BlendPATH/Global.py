"""
Global constants
"""

T_FIXED = 15 + 273.15  # Constant temperature at 15C
STEEL_RHO_KG_M3 = 7_840

# Conversions
MPA2PA = 1_000_000
BAR2PA = 100_000
MM2M = 1 / 1_000
KM2M = 1_000
MM2IN = 0.0393701
KM2MI = 0.621371
KW2W = 1_000
MJ2J = 1e6
W2MW = 1 / 1e6
MW2HP = 1_341.02
MJ2MMBTU = 1 / 1_055.06
S2DAY = 60 * 60 * 24
DAY2HR = 24
MW2MMBTUDAY = MJ2MMBTU * 24 * 3_600

# Solver constants
RELAX_FACTOR = 1.5
MAX_ITER = 1000
SOLVER_TOL = 1e-3
MIN_PRES = 20 * BAR2PA

# Parallel loop solver
PL_LEN_TOL = 0.001

# Max segment L/D
SEG_MAX = 27_500
