import numpy as np
import constants as const

def E_1_waveguide(R0, sig, H, alpha_nl, hbar_c, r_e, r_c):
    # Row 1: Optical-magnetic waveguide confinement (refraction on R0 + magnetic pinch on sigma)
    E_refract = const.ALPHA_FS * alpha_nl * (hbar_c / r_c) * (R0 / r_e)
    E_pinch = alpha_nl * hbar_c * np.sqrt(H) * (sig ** 2) / (R0 ** 2)
    return E_refract + E_pinch


def E_2_gradient_kin(R0, sig, m_tor, n_pol, hbar_c):
    # Row 2: Circulation angular momentum (1/R0) + ocean density equalization pressure (R0/sigma^2)
    E_mom = 0.5 * hbar_c * (m_tor + 0.5 * n_pol) / R0
    E_grad_p = 0.5 * hbar_c * R0 / (sig ** 2)
    return E_mom + E_grad_p


def E_3_torsion_buckling(R0, sig, H, beta_nl, eta_visc, hbar_c, r_e):
    # Row 3: Ribbon torsion stiffness (beta_nl) + two-channel shear buckling cascade (eta_visc)
    E_torsion = beta_nl * hbar_c * (H ** 2) * (r_e / (R0 ** 2) + r_e / (sig ** 2))
    shear_rad = eta_visc * (H ** 2) * (r_e / R0)
    shear_tan = eta_visc * H * (r_e / sig)
    chi = shear_rad + shear_tan
    buckling_factor = 1.0 + (chi ** 3) / (1.0 + 1e-4 * (chi ** 3))
    return E_torsion / buckling_factor



def E_4_compaction(R0, sig, H, gamma_nl, delta_pack, hbar_c, r_e):
    # Row 4: Two-channel compaction wall ("pores closed": radial core + tangential turn-to-turn packing)
    r_core = r_e * np.sqrt(np.maximum(gamma_nl, 1e-6))
    phi_rad = (r_core / R0) ** 2
    phi_tan = delta_pack * H * (r_e ** 2) / (R0 * sig)

    phi_total = phi_rad + phi_tan
    phi_clamped = np.minimum(phi_total, 0.999)
    excess = np.maximum(phi_total - 0.999, 0.0)
    wall = (phi_clamped / (1.0 - phi_clamped)) + 1e6 * excess
    return 0.1 * hbar_c * wall / r_e



def soliton_total_energy_mev(
    R0_fm, sigma_fm, m_tor=1, n_pol=1,
    alpha_nl=0.15, beta_nl=1.0, gamma_nl=0.05, eta_visc=1.0, delta_pack=1.0
):
    hbar_c = (const.HBAR * const.C / const.MEV_TO_J) * 1e15
    r_e = const.R_CLASSICAL_E * 1e15
    r_c = const.R_COMPTON_E * 1e15

    R0 = np.maximum(R0_fm, 1e-3)
    sig = np.maximum(sigma_fm, 1e-3)
    H = m_tor * n_pol

    E1 = E_1_waveguide(R0, sig, H, alpha_nl, hbar_c, r_e, r_c)
    E2 = E_2_gradient_kin(R0, sig, m_tor, n_pol, hbar_c)
    E3 = E_3_torsion_buckling(R0, sig, H, beta_nl, eta_visc, hbar_c, r_e)
    E4 = E_4_compaction(R0, sig, H, gamma_nl, delta_pack, hbar_c, r_e)

    return E1 + E2 + E3 + E4


def compute_observables(
    R0_fm, sigma_fm, m_tor=1, n_pol=1,
    alpha_nl=0.15, beta_nl=1.0, gamma_nl=0.05, eta_visc=1.0, delta_pack=1.0
):
    """
    Computes equilibrium rest mass (MeV), topological Hopf charge, spin,
    and effective g-factor from variational soliton parameters.
    """
    mass_mev = soliton_total_energy_mev(
        R0_fm, sigma_fm, m_tor, n_pol, alpha_nl, beta_nl, gamma_nl, eta_visc, delta_pack
    )

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
