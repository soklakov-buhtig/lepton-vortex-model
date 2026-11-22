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

R, Z, psi, obs = solver.solve_stationary_vortex(m_tor=m_tor, n_pol=n_pol, alpha_nl=alpha_nl)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Meridional Cross-Section (r, z) of Toroidal Vortex Core")
    fig = go.Figure(
        data=go.Contour(
            x=R[:, 0],
            y=Z[0, :],
            z=psi.T,
            colorscale="Viridis",
            colorbar=dict(title="Field Amplitude"),
        )
    )
    fig.update_layout(
        xaxis_title="Radial coordinate r / R_Compton_e",
        yaxis_title="Axial coordinate z / R_Compton_e",
        yaxis=dict(scaleanchor="x", scaleratio=1, range=[-5, 5]),
        xaxis=dict(range=),
        height=520,
    )

   
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Solver Status & Observables")
    st.info("Status: Initial ansatz scaffold active. PDE relaxation solver pending.")
    st.write("**Computed Observables (Placeholder):**")
    st.json(obs)
