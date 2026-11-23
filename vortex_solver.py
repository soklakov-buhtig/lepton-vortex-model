import numpy as np
from scipy.optimize import minimize, differential_evolution
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
    eta_visc=1.0,
    delta_pack=1.0,
):
    """
    Finds equilibrium toroidal geometry (R0*, sigma*) by minimizing total soliton energy
    for given topological numbers (m_tor, n_pol) and global medium constants (alpha_nl, beta_nl, gamma_nl).
    """
    r_c_m = const.R_COMPTON_MU if lepton == "muon" else const.R_COMPTON_E
    r_c_fm = r_c_m * 1e15

    r0_init, sig_init, _ = _screen_2d_minimum(m_tor, n_pol, alpha_nl, beta_nl, gamma_nl, eta_visc, delta_pack)
    x0 = np.array([np.log(r0_init), np.log(sig_init)])


    def objective(x):
        R0_try = np.exp(x[0])
        sig_try = np.exp(x[1])
        return phys.soliton_total_energy_mev(
            R0_try, sig_try, m_tor=m_tor, n_pol=n_pol,
            alpha_nl=alpha_nl, beta_nl=beta_nl, gamma_nl=gamma_nl, eta_visc=eta_visc, delta_pack=delta_pack
        )

    res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 1500, "xatol": 1e-6, "fatol": 1e-6})
    R0_opt = float(np.exp(res.x[0]))
    sig_opt = float(np.exp(res.x[1]))

    effective_scale_fm = max(R0_opt + 2.5 * sig_opt, 0.25 * r_c_fm)
    R, Z, _ = build_lepton_grid(effective_scale_fm * 1e-15, span_factors=1.5)

    rho_torus = np.sqrt((R - R0_opt) ** 2 + Z ** 2)
    psi = np.exp(-0.5 * (rho_torus / sig_opt) ** 2) * (R / (R + 0.05 * R0_opt))

    obs = phys.compute_observables(R0_opt, sig_opt, m_tor, n_pol, alpha_nl, beta_nl, gamma_nl, eta_visc, delta_pack)
    obs["R_compton_fm"] = float(r_c_fm)
    obs["plot_scale_fm"] = float(1.5 * effective_scale_fm / 4.0)
    return R, Z, psi, obs

def _screen_2d_minimum(m_tor, n_pol, alpha_nl, beta_nl, gamma_nl, eta_visc, delta_pack=1.0):
    grid_1d = np.geomspace(0.05, 15000.0, 70)
    R_g, S_g = np.meshgrid(grid_1d, grid_1d, indexing="ij")
    E_g = phys.soliton_total_energy_mev(R_g, S_g, m_tor, n_pol, alpha_nl, beta_nl, gamma_nl, eta_visc, delta_pack)
    idx = np.unravel_index(np.argmin(E_g), E_g.shape)
    return float(R_g[idx]), float(S_g[idx]), float(E_g[idx])


def _fast_mass_eval(m_tor, n_pol, alpha_nl, beta_nl, gamma_nl, eta_visc, delta_pack=1.0):
    r0_init, sig_init, _ = _screen_2d_minimum(m_tor, n_pol, alpha_nl, beta_nl, gamma_nl, eta_visc, delta_pack=delta_pack)
    x0 = np.array([np.log(r0_init), np.log(sig_init)])

    def obj(x):
        r_log, s_log = x
        return phys.soliton_total_energy_mev(
            np.exp(r_log), np.exp(s_log), m_tor, n_pol, alpha_nl, beta_nl, gamma_nl, eta_visc, delta_pack=delta_pack
        )

    res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 250, "xatol": 1e-4, "fatol": 1e-4})
    return float(res.fun)


def calibrate_medium_constants(m_tor_e, n_pol_e, m_tor_mu, n_pol_mu):
    bounds = [
        const.MEDIUM_BOUNDS["alpha_nl"],
        const.MEDIUM_BOUNDS["beta_nl"],
        const.MEDIUM_BOUNDS["gamma_nl"],
        const.MEDIUM_BOUNDS["eta_visc"],
        const.MEDIUM_BOUNDS["delta_pack"],
    ]


    log_bounds = [(np.log10(lo), np.log10(hi)) for lo, hi in bounds]

    def loss(log_params):
        a, b, g, eta, d_pack = 10.0 ** np.array(log_params)
        me = _fast_mass_eval(m_tor_e, n_pol_e, a, b, g, eta, d_pack)
        mmu = _fast_mass_eval(m_tor_mu, n_pol_mu, a, b, g, eta, d_pack)
        return np.log(max(me, 1e-6) / const.M_ELECTRON_MEV) ** 2 + np.log(max(mmu, 1e-6) / const.M_MUON_MEV) ** 2

    res = differential_evolution(loss, log_bounds, maxiter=40, popsize=8, seed=42, polish=True)
    return [float(10.0 ** v) for v in res.x], float(res.fun)
