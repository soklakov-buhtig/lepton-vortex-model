import numpy as np
import streamlit as st
import plotly.graph_objects as go
import constants as const
import vortex_solver as solver

st.set_page_config(page_title="Lepton Vortex Model (Preprint III)", layout="wide")

st.title("Helical-Toroidal Soliton Model of Electron & Muon")
st.caption(
    f"Microscale continuum modeling of the non-fermionic medium | "
    f"Builds on Preprint I (doi:{const.PREPRINT_I_DOI}) & Preprint II (doi:{const.PREPRINT_II_DOI})"
)

with st.sidebar:

    st.header("1. Global Medium Properties")
    for k, default in [("alpha_nl", 0.25), ("beta_nl", 1.0), ("gamma_nl", 0.05), ("eta_visc", 1.2)]:
        
        if k not in st.session_state:
            st.session_state[k] = default

    if "pending_calib" in st.session_state:
        a, b, g, eta = st.session_state.pop("pending_calib")
        st.session_state["alpha_nl"] = round(a, 3)
        st.session_state["beta_nl"] = round(b, 3)
        st.session_state["gamma_nl"] = round(g, 4)
        st.session_state["eta_visc"] = round(eta, 2)


    for k, default in [("alpha_nl", 0.25), ("beta_nl", 1.0), ("gamma_nl", 0.05), ("eta_visc", 1.2)]:
        if k not in st.session_state:
            st.session_state[k] = default

    alpha_nl = st.slider("EM saturation coupling (alpha_nl)", 0.01, 50.0, key="alpha_nl", step=0.05)
    beta_nl = st.slider("Faddeev-Skyrme knot stiffness (beta_nl)", 0.01, 10.0, key="beta_nl", step=0.05)
    gamma_nl = st.slider("Background medium pressure (gamma_nl)", 0.001, 2.0, key="gamma_nl", step=0.005)
    eta_visc = st.slider("Medium shear viscosity (eta_visc)", 0.1, 20.0, key="eta_visc", step=0.1)

    if st.button("Run Global Medium Calibrator", use_container_width=True):
        with st.spinner("Calibrating medium constants via Differential Evolution..."):
            best_p, loss_val = solver.calibrate_medium_constants(
                st.session_state.get("m_tor_e", 2),
                st.session_state.get("n_pol_e", 3),
                st.session_state.get("m_tor_mu", 5),
                st.session_state.get("n_pol_mu", 7),
            )
            st.session_state["pending_calib"] = best_p
            st.session_state["calib_loss"] = loss_val
            st.rerun()


    if "calib_loss" in st.session_state:
        st.caption(f"Calibrator log-error: `{st.session_state['calib_loss']:.5f}`")

    st.divider()
    st.header("2. Electron Topology")
    m_tor_e = st.number_input("Electron toroidal winding (m_tor)", min_value=1, max_value=20, value=2, key="m_tor_e")
    n_pol_e = st.number_input("Electron poloidal winding (n_pol)", min_value=1, max_value=20, value=3, key="n_pol_e")

    st.divider()
    st.header("3. Muon Topology")
    m_tor_mu = st.number_input("Muon toroidal winding (m_tor)", min_value=1, max_value=20, value=5, key="m_tor_mu")
    n_pol_mu = st.number_input("Muon poloidal winding (n_pol)", min_value=1, max_value=20, value=7, key="n_pol_mu")

    st.divider()
    st.subheader("Target Experimental Benchmarks")
    st.metric("Electron Rest Mass", f"{const.M_ELECTRON_MEV:.6f} MeV")
    st.metric("Muon Rest Mass", f"{const.M_MUON_MEV:.4f} MeV")
    st.metric("Target Ratio (m_mu / m_e)", f"{const.MUON_ELECTRON_MASS_RATIO:.4f}")


def make_vortex_figure(R, Z, psi, r_c_fm, colorscale="Viridis"):
    r_max = 4.0 * r_c_fm
    z_max = 2.0 * r_c_fm
    fig = go.Figure(
        data=go.Contour(
            x=R[:, 0],
            y=Z[0, :],
            z=psi.T,
            colorscale=colorscale,
            colorbar=dict(title="Field"),
        )
    )
    fig.update_layout(
        xaxis_title="Radial r [fm]",
        yaxis_title="Axial z [fm]",
        xaxis=dict(range=list((0, r_max)), constrain="domain"),
        yaxis=dict(scaleanchor="x", scaleratio=1, range=list((-z_max, z_max)), constrain="domain"),
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def make_3d_helical_torus(R0, sigma, m_tor, n_pol, line_color="#00f2fe"):
    # Transparent reference torus shell
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, 2 * np.pi, 24)
    U, V = np.meshgrid(u, v)
    X_tor = (R0 + sigma * np.cos(V)) * np.cos(U)
    Y_tor = (R0 + sigma * np.cos(V)) * np.sin(U)
    Z_tor = sigma * np.sin(V)

    # Helical ribbon centerline + normal twist along the torus
    t = np.linspace(0, 2 * np.pi, 600)
    w = np.linspace(-1.0, 1.0, 6)
    T, W = np.meshgrid(t, w, indexing="ij")

    phi = m_tor * T
    theta = n_pol * T
    ribbon_half_w = 0.28 * sigma

    # Center curve on the torus
    r_c = R0 + sigma * np.cos(theta)
    xc, yc, zc = r_c * np.cos(phi), r_c * np.sin(phi), sigma * np.sin(theta)

    # Ribbon width vector lying tangent to the torus surface (twisting with theta)
    nx = -np.sin(theta) * np.cos(phi)
    ny = -np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)

    Xr = xc + W * ribbon_half_w * nx
    Yr = yc + W * ribbon_half_w * ny
    Zr = zc + W * ribbon_half_w * nz

    fig = go.Figure()
    fig.add_trace(go.Surface(x=X_tor, y=Y_tor, z=Z_tor, opacity=0.14, showscale=False, colorscale="Greys"))
    fig.add_trace(go.Surface(x=Xr, y=Yr, z=Zr, showscale=False, colorscale=[[0, line_color], [1, "#ffffff"]]))

    lim = R0 + 1.5 * sigma
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=list((-lim, lim)), title="X [fm]"),
            yaxis=dict(range=list((-lim, lim)), title="Y [fm]"),
            zaxis=dict(range=list((-lim, lim)), title="Z [fm]"),
            aspectmode="cube",
        ),
        height=400,
        margin=dict(l=0, r=0, t=10, b=10),
        showlegend=False,
    )
    return fig



# --- Top Row: Electron ---
R_e, Z_e, psi_e, meta_e = solver.solve_variational_lepton(
    "electron", m_tor=m_tor_e, n_pol=n_pol_e,
    alpha_nl=alpha_nl, beta_nl=beta_nl, gamma_nl=gamma_nl, eta_visc=eta_visc
)
st.subheader(f"1. Electron Vortex Core — Mode ({m_tor_e}, {n_pol_e}) [R_Compton = {meta_e['R_compton_fm']:.2f} fm]")
col_e1, col_e2, col_e3 = st.columns([1.15, 1.15, 0.8])
with col_e1:
    st.plotly_chart(make_vortex_figure(R_e, Z_e, psi_e, meta_e["plot_scale_fm"], "Viridis"), use_container_width=True)
with col_e2:
    if st.toggle("Render 3D Electron Helix", key="render_3d_e"):
        st.plotly_chart(
            make_3d_helical_torus(meta_e["R0_fm"], meta_e["sigma_fm"], m_tor_e, n_pol_e, "#00f2fe"),
            use_container_width=True,
        )
    else:
        st.info("3D rendering paused to save resources. Toggle switch above to inspect the helical-toroidal core.")
with col_e3:
    st.metric("Computed Electron Mass", f"{meta_e['mass_mev']:.5f} MeV", delta=f"Target: {const.M_ELECTRON_MEV:.5f}")
    st.metric("Major Radius R0 / Width σ", f"{meta_e['R0_fm']:.1f} / {meta_e['sigma_fm']:.1f} fm")
    st.write(f"**Hopf charge $Q_H$:** `{meta_e['hopf_charge']}` | **$g$-factor:** `{meta_e['g_factor']:.6f}`")

st.divider()

# --- Bottom Row: Muon ---
R_mu, Z_mu, psi_mu, meta_mu = solver.solve_variational_lepton(
    "muon", m_tor=m_tor_mu, n_pol=n_pol_mu,
    alpha_nl=alpha_nl, beta_nl=beta_nl, gamma_nl=gamma_nl, eta_visc=eta_visc
)

st.subheader(f"2. Muon Vortex Core — Mode ({m_tor_mu}, {n_pol_mu}) [R_Compton = {meta_mu['R_compton_fm']:.3f} fm]")
col_mu1, col_mu2, col_mu3 = st.columns([1.15, 1.15, 0.8])
with col_mu1:
    st.plotly_chart(make_vortex_figure(R_mu, Z_mu, psi_mu, meta_mu["plot_scale_fm"], "Plasma"), use_container_width=True)
with col_mu2:
    if st.toggle("Render 3D Muon Helix", key="render_3d_mu"):
        st.plotly_chart(
            make_3d_helical_torus(meta_mu["R0_fm"], meta_mu["sigma_fm"], m_tor_mu, n_pol_mu, "#ff4b4b"),
            use_container_width=True,
        )
    else:
        st.info("3D rendering paused to save resources. Toggle switch above to inspect the helical-toroidal core.")
with col_mu3:
    ratio_calc = meta_mu["mass_mev"] / max(meta_e["mass_mev"], 1e-12)
    st.metric("Computed Muon Mass", f"{meta_mu['mass_mev']:.3f} MeV", delta=f"Target: {const.M_MUON_MEV:.2f}")
    st.metric("Mass Ratio (m_mu / m_e)", f"{ratio_calc:.2f}", delta=f"Target: {const.MUON_ELECTRON_MASS_RATIO:.2f}")
    st.write(f"**R0 / σ:** `{meta_mu['R0_fm']:.2f} / {meta_mu['sigma_fm']:.2f} fm` | **$Q_H$:** `{meta_mu['hopf_charge']}`")
