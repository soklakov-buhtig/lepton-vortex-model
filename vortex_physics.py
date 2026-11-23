import numpy as np
import constants as const


def soliton_total_energy_mev(R0_fm, sigma_fm, m_tor=1, n_pol=1, alpha_nl=0.15, beta_nl=1.0, gamma_nl=0.05):
    hbar_c = (const.HBAR * const.C / const.MEV_TO_J) * 1e15
    R0 = max(R0_fm, 1e-4)
    sig = max(sigma_fm, 1e-4)

    H = m_tor * n_pol
    r_c_e = const.R_COMPTON_E * 1e15

    # 1. Angular momentum / circulation kinetic energy (~ 1/R0, grows when core shrinks)
    E_kin = 0.5 * hbar_c * (m_tor + 0.5 * n_pol) / R0

    # 2. Tube cross-section confinement (balances sigma relative to R0)
    E_tube = 0.5 * hbar_c * R0 / (sig ** 2)

    # 3. Supercoiling ribbon contraction (pulls R0 inward as H^3 grows)
    u_rel = (H ** 3) * ((r_c_e / R0) ** 2)
    n_eff = 1.0 + (alpha_nl * u_rel) / (1.0 + gamma_nl * u_rel)
    E_supercoil = (hbar_c / r_c_e) * (R0 / r_c_e) * (H ** 2) * n_eff

    # 4. Short-range ribbon layer repulsion (prevents collapse to R0 -> 0)
    E_core_repulsion = beta_nl * hbar_c * (r_c_e / (R0 ** 2)) * (1.0 + 0.1 * H)

    return E_kin + E_tube + E_supercoil + E_core_repulsion


def compute_observables(R0_fm, sigma_fm, m_tor=1, n_pol=1, alpha_nl=0.15, beta_nl=1.0, gamma_nl=0.05):
    """
    Computes equilibrium rest mass (MeV), topological Hopf charge, spin,
    and effective g-factor from variational soliton parameters.
    """
    mass_mev = soliton_total_energy_mev(R0_fm, sigma_fm, m_tor, n_pol, alpha_nl, beta_nl, gamma_nl)
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
