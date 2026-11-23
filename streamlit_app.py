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
    alpha_nl = st.slider("EM saturation coupling (alpha_nl)", 0.0, 2.0, 0.15, step=0.05)
    beta_nl = st.slider("Faddeev-Skyrme knot stiffness (beta_nl)", 0.0, 5.0, 1.0, step=0.1)
    gamma_nl = st.slider("Background medium pressure (gamma_nl)", 0.0, 1.0, 0.05, step=0.01)


    st.divider()
    st.header("2. Electron Topology")
    m_tor_e = st.number_input("Electron toroidal winding (m_tor)", min_value=1, max_value=20, value=1)
    n_pol_e = st.number_input("Electron poloidal winding (n_pol)", min_value=1, max_value=20, value=1)

    st.divider()
    st.header("3. Muon Topology")
    m_tor_mu = st.number_input("Muon toroidal winding (m_tor)", min_value=1, max_value=20, value=2)
    n_pol_mu = st.number_input("Muon poloidal winding (n_pol)", min_value=1, max_value=20, value=3)

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
    # Transparent reference torus surface
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, 2 * np.pi, 28)
    U, V = np.meshgrid(u, v)
    X_tor = (R0 + sigma * np.cos(V)) * np.cos(U)
    Y_tor = (R0 + sigma * np.cos(V)) * np.sin(U)
    Z_tor = sigma * np.sin(V)

    # Helical vortex filament winding (m_tor, n_pol)
    t = np.linspace(0, 2 * np.pi, 1200)
    phi = m_tor * t
    theta = n_pol * t
    x_line = (R0 + sigma * np.cos(theta)) * np.cos(phi)
    y_line = (R0 + sigma * np.cos(theta)) * np.sin(phi)
    z_line = sigma * np.sin(theta)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=X_tor,
            y=Y_tor,
            z=Z_tor,
            opacity=0.18,
            showscale=False,
            colorscale="Greys",
            name="Core Shell",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x_line,
            y=y_line,
            z=z_line,
            mode="lines",
            line=dict(color=line_color, width=5),
            name=f"Helix ({m_tor},{n_pol})",
        )
    )
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
    alpha_nl=alpha_nl, beta_nl=beta_nl, gamma_nl=gamma_nl
)
st.subheader(f"1. Electron Vortex Core — Mode ({m_tor_e}, {n_pol_e}) [R_Compton = {meta_e['R_compton_fm']:.2f} fm]")
col_e1, col_e2, col_e3 = st.columns([1.15, 1.15, 0.8])
with col_e1:
    st.plotly_chart(make_vortex_figure(R_e, Z_e, psi_e, meta_e["R_compton_fm"], "Viridis"), use_container_width=True)
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
    alpha_nl=alpha_nl, beta_nl=beta_nl, gamma_nl=gamma_nl
)

st.subheader(f"2. Muon Vortex Core — Mode ({m_tor_mu}, {n_pol_mu}) [R_Compton = {meta_mu['R_compton_fm']:.3f} fm]")
col_mu1, col_mu2, col_mu3 = st.columns([1.15, 1.15, 0.8])
with col_mu1:
    st.plotly_chart(make_vortex_figure(R_mu, Z_mu, psi_mu, meta_mu["R_compton_fm"], "Plasma"), use_container_width=True)
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
