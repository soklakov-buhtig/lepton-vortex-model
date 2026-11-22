import numpy as np

# =====================================================================
# 1. Fundamental Physical Constants (SI & Natural Units, CODATA)
# =====================================================================
C = 2.99792458e8               # Speed of light in vacuum [m/s]
HBAR = 1.054571817e-34         # Reduced Planck constant [J*s]
E_CHARGE = 1.602176634e-19     # Elementary electric charge [C]
EPS0 = 8.8541878128e-12        # Vacuum permittivity [F/m]
MU0 = 4.0 * np.pi * 1e-7       # Vacuum permeability [H/m]
ALPHA_FS = 7.2973525693e-3     # Fine-structure constant (~1/137.036)

# Conversion factors
EV_TO_J = 1.602176634e-19
MEV_TO_J = 1.602176634e-13
MEV_TO_KG = MEV_TO_J / (C ** 2)

# =====================================================================
# 2. Target Lepton Benchmarks (Electron & Muon)
# =====================================================================
# Electron (1st generation lepton)
M_ELECTRON_MEV = 0.51099895000          # Rest mass energy [MeV]
M_ELECTRON_KG = M_ELECTRON_MEV * MEV_TO_KG
R_COMPTON_E = HBAR / (M_ELECTRON_KG * C)  # Reduced Compton wavelength (~3.8616e-13 m)
R_CLASSICAL_E = E_CHARGE**2 / (4.0 * np.pi * EPS0 * M_ELECTRON_KG * C**2)  # (~2.8179e-15 m)
G_FACTOR_E = 2.00231930436256           # Electron g-factor

# Muon (2nd generation lepton)
M_MUON_MEV = 105.6583755                # Rest mass energy [MeV]
M_MUON_KG = M_MUON_MEV * MEV_TO_KG
R_COMPTON_MU = HBAR / (M_MUON_KG * C)   # Reduced Compton wavelength (~1.8676e-15 m)
G_FACTOR_MU = 2.0023318418              # Muon g-factor

# Key dimensionless mass ratio benchmark (main target for excited topological mode)
MUON_ELECTRON_MASS_RATIO = M_MUON_MEV / M_ELECTRON_MEV  # ~206.768283

# =====================================================================
# 3. Non-Fermionic Medium Reference Scales (Links to Preprints I & II)
# =====================================================================
PREPRINT_I_DOI = "10.5281/zenodo.23952161"
PREPRINT_II_DOI = "10.5281/zenodo.24446953"

# Default numerical grid placeholders (in units of R_COMPTON_E)
DEFAULT_R_MAX = 10.0
DEFAULT_Z_MAX = 10.0
DEFAULT_GRID_N = 128
