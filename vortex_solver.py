import numpy as np
import constants as const
import vortex_physics as phys


def build_lepton_grid(r_compton_m, span_factors=4.0, n_points=const.DEFAULT_GRID_N):
    """
    Creates an axisymmetric cylindrical grid (r, z) in femtometers (1e-15 m)
    anchored directly to the particle's reduced Compton wavelength from constants.py.
    """
    r_compton_fm = r_compton_m * 1e15
    r_max = span_factors * r_compton_fm
    z_max = 0.5 * r_max

    r = np.linspace(1e-4 * r_compton_fm, r_max, n_points)
    z = np.linspace(-z_max, z_max, n_points)
    R, Z = np.meshgrid(r, z, indexing="ij")
    return R, Z, r_compton_fm


def toroidal_profile_from_constants(R, Z, r_compton_fm, m_tor=1, n_pol=1, alpha_nl=0.1):
    """
    Constructs a physically scaled toroidal vortex profile anchored to R_COMPTON
    and fine-structure coupling ALPHA_FS, modulated by topological numbers (m_tor, n_pol).
    """
    # Major radius anchored to Compton scale; core tube thickness scales with alpha_fs and winding
    R0 = r_compton_fm / np.sqrt(m_tor)
    sigma = r_compton_fm * np.sqrt(const.ALPHA_FS) * (1.0 + 0.15 * n_pol) / (1.0 + 0.3 * alpha_nl)

    rho_torus = np.sqrt((R - R0) ** 2 + Z ** 2)
    psi = np.exp(-0.5 * (rho_torus / sigma) ** 2) * (R / (R + 0.1 * r_compton_fm))
    return psi, R0, sigma


def get_lepton_ansatz(lepton="electron", m_tor=1, n_pol=1, alpha_nl=0.1):
    """
    Returns physically scaled (R, Z) grid in femtometers and field profile for 'electron' or 'muon'.
    """
    if lepton == "muon":
        r_c = const.R_COMPTON_MU
    else:
        r_c = const.R_COMPTON_E

    R, Z, r_c_fm = build_lepton_grid(r_c)
    psi, R0, sigma = toroidal_profile_from_constants(R, Z, r_c_fm, m_tor, n_pol, alpha_nl)
    meta = {"R_compton_fm": r_c_fm, "R0_fm": R0, "sigma_fm": sigma}
    return R, Z, psi, meta
