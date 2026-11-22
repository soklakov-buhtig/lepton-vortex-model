# lepton-vortex-model
Numerical modeling of electron and muon as helical-toroidal solitons in a nonlinear non-fermionic medium (Preprint III)
# Lepton Vortex Model: Helical-Toroidal Solitons in a Non-Fermionic Medium (Preprint III)

[![Streamlit App](https://streamlit.io)](https://streamlit.io)
[![DOI Preprint II](https://shields.io)](https://doi.org)
[![DOI Preprint I](https://shields.io)](https://doi.org)

Numerical modeling framework for **Preprint III**, investigating elementary leptons (the **electron** and **muon**) as stable, self-trapped helical-toroidal vortex solitons $(m_{\mathrm{tor}}, n_{\mathrm{pol}})$ within a nonlinear elastoplastic **non-fermionic energy medium**.

---

### Research Context (Three-Scale Continuum Program)
* **Macroscale (Preprint I):** Isotropic scalar medium pressure $S$ reproduces the exterior Schwarzschild metric and relativistic tests in flat spacetime ([doi:10.5281/zenodo.23952161](https://doi.org)).
* **Megascale (Preprint II):** Trace-free shear stress $\pi_{\mu\nu}$ explains galactic rotation curves across 175 SPARC galaxies without dark matter halos ([doi:10.5281/zenodo.24446953](https://doi.org)).
* **Microscale (Preprint III — Active):** Nonlinear strong-field self-trapping and topological winding modes $(m_{\mathrm{tor}}, n_{\mathrm{pol}})$ aim to derive lepton rest masses ($m_\mu / m_e \approx 206.768$), quantized charge, half-integer spin, and magnetic moments from first-principles continuum equations.

### Repository Structure
* `constants.py` — Fundamental CODATA constants and experimental lepton benchmarks.
* `vortex_physics.py` — Constitutive laws of the nonlinear medium, energy density functional, and integral observables.
* `vortex_solver.py` — Axisymmetric $(r, z)$ cylindrical grid generator and nonlinear PDE relaxation solver.
* `streamlit_app.py` — Interactive meridional cross-section visualizer and mode explorer.

### Author
* **Alexander V. Soklakov** — Independent Researcher ([ORCID: 0009-0002-4586-1800](https://orcid.org))
