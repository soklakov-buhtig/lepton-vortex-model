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
    st.header("Topological & Medium Parameters")
    m_tor = st.number_input("Toroidal winding number (m_tor)", min_value=1, max_value=10, value=1)
    n_pol = st.number_input("Poloidal winding number (n_pol)", min_value=1, max_value=10, value=1)
    alpha_nl = st.slider("Nonlinear stiffness coupling (alpha_nl)", 0.0, 2.0, 0.1, step=0.05)

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
        xaxis_title="Radial coordinate r [fm]",
        yaxis_title="Axial coordinate z [fm]",
        xaxis=dict(range=list((0, r_max)), constrain="domain"),
        yaxis=dict(scaleanchor="x", scaleratio=1, range=list((-z_max, z_max)), constrain="domain"),
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# --- Top Plot: Electron ---
R_e, Z_e, psi_e, meta_e = solver.get_lepton_ansatz("electron", m_tor=m_tor, n_pol=n_pol, alpha_nl=alpha_nl)
st.subheader(f"1. Electron Vortex Core (R_Compton = {meta_e['R_compton_fm']:.2f} fm)")
col_e1, col_e2 = st.columns([2, 1])
with col_e1:
    st.plotly_chart(make_vortex_figure(R_e, Z_e, psi_e, meta_e["R_compton_fm"], "Viridis"), use_container_width=True)
with col_e2:
    st.markdown("**Electron Geometric Anchors (`constants.py`):**")
    st.write(f"- Rest mass: `{const.M_ELECTRON_MEV:.6f} MeV`")
    st.write(f"- Major torus radius $R_0$: `{meta_e['R0_fm']:.2f} fm`")
    st.write(f"- Core tube width $\\sigma$: `{meta_e['sigma_fm']:.2f} fm`")
    st.write(f"- Classical radius $r_e$: `{const.R_CLASSICAL_E * 1e15:.3f} fm`")

st.divider()

# --- Bottom Plot: Muon ---
R_mu, Z_mu, psi_mu, meta_mu = solver.get_lepton_ansatz("muon", m_tor=m_tor, n_pol=n_pol, alpha_nl=alpha_nl)
st.subheader(f"2. Muon Vortex Core (R_Compton = {meta_mu['R_compton_fm']:.3f} fm)")
col_mu1, col_mu2 = st.columns([2, 1])
with col_mu1:
    st.plotly_chart(make_vortex_figure(R_mu, Z_mu, psi_mu, meta_mu["R_compton_fm"], "Plasma"), use_container_width=True)
with col_mu2:
    st.markdown("**Muon Geometric Anchors (`constants.py`):**")
    st.write(f"- Rest mass: `{const.M_MUON_MEV:.4f} MeV`")
    st.write(f"- Major torus radius $R_0$: `{meta_mu['R0_fm']:.3f} fm`")
    st.write(f"- Core tube width $\\sigma$: `{meta_mu['sigma_fm']:.3f} fm`")
    st.write(f"- Scale compression ($R_e / R_\\mu$): `{const.MUON_ELECTRON_MASS_RATIO:.2f}x`")
