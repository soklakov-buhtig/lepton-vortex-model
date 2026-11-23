import numpy as np
import constants as const


def soliton_total_energy_mev(R0_fm, sigma_fm, m_tor=1, n_pol=1, alpha_nl=0.15, beta_nl=1.0, gamma_nl=0.05):
    """
    Computes total rest energy (mass in MeV) of a helical-toroidal soliton
    with major radius R0_fm and tube width sigma_fm in a nonlinear non-fermionic medium.

    Four physical energy terms (in natural hbar*c = 197.327 MeV*fm units):
      1. Line tension & helical twist energy (elastic stretching along torus + poloidal spin)
      2. Faddeev-Skyrme topological knotting stiffness (proportional to beta_nl * (m_tor * n_pol)^2)
      3. Nonlinear electromagnetic self-energy with saturation (governed by alpha_fs & alpha_nl)
      4. Isotropic background medium pressure work (governed by gamma_nl, link to Preprint I)
    """
    hbar_c_mev_fm = (const.HBAR * const.C / const.MEV_TO_J) * 1e15  # ~197.32698 MeV*fm
    R0 = max(R0_fm, 1e-4)
    sig = max(sigma_fm, 1e-4)

    # Topological knot complexity index (Hopf-like invariant H = m_tor * n_pol)
    H_knot = m_tor * n_pol
    r_c_e = const.R_COMPTON_E * 1e15
    u_rel = (H_knot ** 2) * ((r_c_e / R0) ** 2) * ((r_c_e / sig) ** 2)
    n_eff = 1.0 + (alpha_nl * u_rel) / (1.0 + gamma_nl * u_rel)

    # 1. Elastic line tension + helical circulation kinetic energy (~ 1/sig^2 compressed along 2*pi*R0)
    E_elastic = (hbar_c_mev_fm * R0 / (sig ** 2)) * (0.5 * (m_tor ** 2) + 0.5 * (n_pol ** 2) * (sig / R0) ** 2)

    # 2. Faddeev-Skyrme fourth-order gradient stiffness (prevents collapse & drives knot mass hierarchy)
    E_skyrme = beta_nl * hbar_c_mev_fm * (H_knot ** 2) / (R0 * (1.0 + (sig / R0) ** 2))

    # 3. Nonlinear electromagnetic self-repulsion (Coulomb + magnetic dipole, regulated by alpha_nl)
    aspect_log = np.log(1.0 + 8.0 * R0 / sig)
    E_em = (const.ALPHA_FS * hbar_c_mev_fm / (2.0 * np.pi * R0)) * aspect_log / (1.0 + alpha_nl * (const.R_CLASSICAL_E * 1e15 / sig))

    # 4. Volume work against isotropic background medium pressure S_0 (Preprint I coupling)
    torus_volume_fm3 = 2.0 * (np.pi ** 2) * R0 * (sig ** 2)
    E_pressure = gamma_nl * (hbar_c_mev_fm / (const.R_COMPTON_E * 1e15) ** 4) * torus_volume_fm3


    E_elastic_eff = E_elastic / (n_eff ** 0.5)
    E_skyrme_eff = E_skyrme * (n_eff ** 0.5)
    E_em_eff = E_em / n_eff
    E_pressure_eff = E_pressure * (n_eff ** 1.5)
    return E_elastic_eff + E_skyrme_eff + E_em_eff + E_pressure_eff

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
