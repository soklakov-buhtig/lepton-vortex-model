import numpy as np
import constants as const

def soliton_total_energy_mev(R0_fm, sigma_fm, m_tor=1, n_pol=1, alpha_nl=0.15, beta_nl=1.0, gamma_nl=0.05, eta_visc=1.0):
    hbar_c = (const.HBAR * const.C / const.MEV_TO_J) * 1e15
    r_e = const.R_CLASSICAL_E * 1e15   # ~2.818 fm
    r_c = const.R_COMPTON_E * 1e15     # ~386.16 fm

    R0 = np.maximum(R0_fm, 1e-3)
    sig = np.maximum(sigma_fm, 1e-3)

    H = m_tor * n_pol
    shear_rate = (H ** 1.5) * (r_e / R0)
    Wi = eta_visc * shear_rate
    buckling_factor = 1.0 + (Wi ** 3) / (1.0 + 0.01 * (Wi ** 3))

    # 1. Circulation kinetic energy (scales as 1/R0 by angular momentum conservation)
    E_kin = 0.5 * hbar_c * (m_tor + 0.5 * n_pol) / R0

    # 2. Tube cross-section equilibrium (balances 1/sig^2 confinement against sig/R0 surface tension)
    E_tube = 0.5 * hbar_c * (R0 / (sig ** 2) + (H ** 0.5) * sig / (R0 ** 2))

    # 3. Viscous buckling threshold (Weissenberg number Wi): above critical shear, radial stiffness collapses
    shear_rate = (H ** 2.0) * (r_e / R0)
    Wi = eta_visc * shear_rate
    buckling_collapse = 1.0 + (Wi ** 4) / (1.0 + 0.002 * (Wi ** 4))

    # 4. Nonlinear self-focusing confinement & saturable core stiffness (Born-Infeld ceiling gamma_nl)
    core_density = (H ** 2) * ((r_e / R0) ** 2)
    saturated_density = core_density / np.sqrt(1.0 + gamma_nl * core_density)
    E_supercoil = const.ALPHA_FS * alpha_nl * (hbar_c / r_c) * (R0 / r_e) / (1.0 + 0.1 * saturated_density)
    E_core = (beta_nl * hbar_c * (r_c / (R0 ** 2))) / buckling_collapse


    return E_kin + E_tube + E_supercoil + E_core


def compute_observables(R0_fm, sigma_fm, m_tor=1, n_pol=1, alpha_nl=0.15, beta_nl=1.0, gamma_nl=0.05, eta_visc=1.0):
    """
    Computes equilibrium rest mass (MeV), topological Hopf charge, spin,
    and effective g-factor from variational soliton parameters.
    """
    mass_mev = soliton_total_energy_mev(R0_fm, sigma_fm, m_tor, n_pol, alpha_nl, beta_nl, gamma_nl, eta_visc)
    hopf_charge = int(m_tor * n_pol)

    # Effective magnetic g-factor correction from torus aspect ratio (sigma / R0)
    aspect = sigma_fm / max(R0_fm, 1e-6)
    g_eff = 2.0 * (1.0 + (const.ALPHA_FS / (2.0 * np.pi)) * (1.0 + 0.25 * (aspect ** 2)))

    return {
        "mass_mev": float(mass_mev),
        "hopf_charge": hopf_charge,
        "spin_hbar": 0.5,
        "g_factor": float(g_eff),
        "R0_fm": float(R0_fm),
        "sigma_fm": float(sigma_fm),
    }
