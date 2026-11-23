import numpy as np
from scipy.optimize import minimize
import constants as const
import vortex_physics as phys


def build_lepton_grid(r_compton_m, span_factors=4.0, n_points=const.DEFAULT_GRID_N):
    """
    Creates an axisymmetric cylindrical grid (r, z) in femtometers (1e-15 m)
    anchored to the reference Compton scale.
    """
    r_compton_fm = r_compton_m * 1e15
    r_max = span_factors * r_compton_fm
    z_max = 0.5 * r_max

    r = np.linspace(1e-4 * r_compton_fm, r_max, n_points)
    z = np.linspace(-z_max, z_max, n_points)
    R, Z = np.meshgrid(r, z, indexing="ij")
    return R, Z, r_compton_fm


def solve_variational_lepton(
    lepton="electron",
    m_tor=1,
    n_pol=1,
    alpha_nl=0.15,
    beta_nl=1.0,
    gamma_nl=0.05,
):
    """
    Finds equilibrium toroidal geometry (R0*, sigma*) by minimizing total soliton energy
    for given topological numbers (m_tor, n_pol) and global medium constants (alpha_nl, beta_nl, gamma_nl).
    """
    r_c_m = const.R_COMPTON_MU if lepton == "muon" else const.R_COMPTON_E
    r_c_fm = r_c_m * 1e15

    # Initial guess in log-space to guarantee positivity across orders of magnitude
    x0 = np.array([np.log(r_c_fm), np.log(r_c_fm * np.sqrt(const.ALPHA_FS))])

    def objective(x):
        R0_try = np.exp(x[0])
        sig_try = np.exp(x[1])
        return phys.soliton_total_energy_mev(
            R0_try, sig_try, m_tor=m_tor, n_pol=n_pol,
            alpha_nl=alpha_nl, beta_nl=beta_nl, gamma_nl=gamma_nl
        )

    res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 1500, "xatol": 1e-6, "fatol": 1e-6})
    R0_opt = float(np.exp(res.x[0]))
    sig_opt = float(np.exp(res.x[1]))

    # Build grid centered around the equilibrium radius R0_opt
    grid_scale_m = max(R0_opt * 1e-15, 0.25 * r_c_m)
    R, Z, _ = build_lepton_grid(grid_scale_m)

    rho_torus = np.sqrt((R - R0_opt) ** 2 + Z ** 2)
    psi = np.exp(-0.5 * (rho_torus / sig_opt) ** 2) * (R / (R + 0.05 * R0_opt))

    obs = phys.compute_observables(R0_opt, sig_opt, m_tor, n_pol, alpha_nl, beta_nl, gamma_nl)
    obs["R_compton_fm"] = float(r_c_fm)
    return R, Z, psi, obs
