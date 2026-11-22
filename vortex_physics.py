import numpy as np
import constants as const


def nonlinear_permittivity(E_sq, B_sq, alpha_nl=0.0):
    """
    Effective nonlinear response (permittivity/stiffness) of the non-fermionic medium
    in strong-field regions near the vortex core.
    Placeholder for Preprint III constitutive law.
    """
    # TODO: Implement exact nonlinear saturation / self-trapping constitutive relation
    return 1.0 + alpha_nl * (E_sq - B_sq)


def energy_density(r, z, psi, dpsi_dr, dpsi_dz, m_tor=1, n_pol=1):
    """
    Local energy density of a helical-toroidal soliton with topological winding numbers
    (m_tor, n_pol) in cylindrical coordinates (r, z).
    """
    # TODO: Add full electromagnetic + shear-elastic medium energy terms
    grad_sq = dpsi_dr**2 + dpsi_dz**2
    return 0.5 * grad_sq


def compute_observables(r_grid, z_grid, field_solution):
    """
    Integrates total rest mass (energy / c^2), electric charge, magnetic moment,
    and spin angular momentum from a numerical soliton profile.
    """
    # TODO: Implement volume integrals in cylindrical coordinates
    return {
        "mass_mev": 0.0,
        "charge_e": -1.0,
        "spin_hbar": 0.5,
        "mu_bohr": 1.0,
    }
