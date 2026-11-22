import numpy as np
import constants as const
import vortex_physics as phys


def build_cylindrical_grid(r_max=const.DEFAULT_R_MAX, z_max=const.DEFAULT_Z_MAX, n_points=const.DEFAULT_GRID_N):
    """
    Creates a 2D axisymmetric cylindrical grid (r, z) in dimensionless units
    normalized to the electron reduced Compton radius R_COMPTON_E.
    """
    r = np.linspace(1e-4, r_max, n_points)
    z = np.linspace(-z_max, z_max, n_points)
    R, Z = np.meshgrid(r, z, indexing="ij")
    return R, Z


def initial_toroidal_ansatz(R, Z, R0=1.0, sigma=0.35):
    """
    Generates a smooth initial toroidal core ansatz centered at major radius R0
    with minor core width sigma for relaxation / shooting solvers.
    """
    rho_torus = np.sqrt((R - R0) ** 2 + Z ** 2)
    psi_init = np.exp(-0.5 * (rho_torus / sigma) ** 2) * (R / (R + 1e-6))
    return psi_init


def solve_stationary_vortex(m_tor=1, n_pol=1, alpha_nl=0.1, n_points=const.DEFAULT_GRID_N):
    """
    Placeholder solver for stationary helical-toroidal soliton modes.
    Returns grid coordinates, field profile, and integrated observables.
    """
    R, Z = build_cylindrical_grid(n_points=n_points)
    psi = initial_toroidal_ansatz(R, Z)

    # TODO: Implement gradient-flow / Newton-Krylov nonlinear PDE relaxation
    observables = phys.compute_observables(R, Z, psi)
    return R, Z, psi, observables
