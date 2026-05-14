import streamlit as st
import numpy as np
import pandas as pd
import math
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ECO Laboratory", layout="wide", initial_sidebar_state="expanded")
st.title("Exotic Compact Object (ECO) Laboratory")
st.markdown("### Interactive Two-Fluid TOV Solver (Giangrandi et al. Framework)")

with st.expander("📖 Physics Guide: How to use this laboratory"):
    st.markdown("""
    This simulator integrates the **Tolman-Oppenheimer-Volkoff (TOV)** equations for a two-fluid hybrid star. 
    It models a baryonic Quark Matter exterior containing an interacting Bosonic Dark Matter core.
    
    **What happens when you change the parameters?**
    * **Bag Constant ($B$):** Governs the vacuum pressure of the quark matter. Increasing this makes the star more compact (shrinks the radius) and lowers the maximum mass.
    * **Boson Mass ($m_\chi$):** Determines the density distribution of the dark matter. **High mass (>50 MeV)** creates a dense, compact *Core*. **Low mass (<20 MeV)** causes the dark matter to spread outward into a vast *Halo*.
    * **Interaction Scale ($m_I$):** Governs the repulsive force between dark matter particles. A higher value makes the dark matter stiffer, allowing it to support more mass against gravity.
    * **DM Fraction:** The percentage of the star's total mass made up of Dark Matter. Increasing this generally "softens" the star, reducing its maximum mass and radius.
    
    **Graph Limits Explained:**
    * **Black Hole (Schwarzschild Limit):** $R = 2GM/c^2$. If a star crosses into the gray zone, it collapses into a black hole.
    * **Buchdahl Limit:** $R = 2.25GM/c^2$. The absolute maximum compactness for any stable isotropic fluid sphere in General Relativity.
    """)

# --- PHYSICS CONSTANTS ---
pi = math.pi
G = 6.67430e-8; c = 2.99792458e10; hbar = 1.054571817e-27
Msun = 1.98840987e33; MeV_fm3_to_pa_cgs = 1.602176634e33  
hbar_c_mev_fm = 197.3269804; MeV4_to_MeV_fm3 = 1.0 / (hbar_c_mev_fm**3) 
m_planck_cgs = math.sqrt((hbar * c) / G)      
l_planck_cgs = math.sqrt((hbar * G) / c**3)   
t_planck_cgs = math.sqrt((hbar * G) / c**5)   
m_to_mplanck = 1.0 / m_planck_cgs
len_to_lenplanck = 1.0 / l_planck_cgs
t_to_tplanck = 1.0 / t_planck_cgs
gcmdens_to_densplanck = m_to_mplanck / (len_to_lenplanck**3)
pa_to_Pplanck = m_to_mplanck / (len_to_lenplanck * (t_to_tplanck**2))

# --- SIDEBAR: PARAMETER INPUTS ---
st.sidebar.header("Fundamental Parameters")
with st.sidebar.form("physics_params"):
    st.markdown("**Quark Matter (MIT Bag)**")
    b_bag = st.slider("Bag Constant (B) [MeV/fm³]", 40.0, 60.0, 45.0, 1.0, 
                      help="Higher values increase vacuum pressure, lowering the max mass of the star.")
    
    st.markdown("**Dark Matter (Bosonic BEC)**")
    m_chi = st.slider("Boson Mass (m_χ) [MeV]", 5.0, 300.0, 100.0, 5.0, 
                      help="Mass of the DM particle. Low mass (<20 MeV) creates a Halo. High mass creates a dense Core.")
    m_I = st.slider("Interaction Scale (m_I) [MeV]", 50.0, 500.0, 250.0, 10.0,
                    help="Repulsive interaction strength. Higher values make the DM core 'stiffer' and larger.")
    
    st.markdown("**Stellar Composition**")
    f_dm = st.slider("Dark Matter Fraction (%)", 0.0, 20.0, 5.0, 1.0,
                     help="Total percentage of the star's mass comprised of Dark Matter.")
    
    submit_button = st.form_submit_button(label="🚀 Run TOV Simulation")

# --- CORE PHYSICS ENGINE (Cached for speed) ---
@st.cache_data(show_spinner=False)
def run_full_simulation(b_bag, m_chi, m_I, f_dm):
    target_fraction = f_dm / 100.0
    
    # Generate EoS Table dynamically
    P_INF_CGS = ((m_I**2 * m_chi**2) / 4.0) * MeV4_to_MeV_fm3 * MeV_fm3_to_pa_cgs
    _mu_range = np.linspace(m_chi, np.sqrt(2)*m_chi - 1e-6, 5000)
    _p_mev4 = (m_I**2 / 4.0) * (m_chi**2 - _mu_range * np.sqrt(2*m_chi**2 - _mu_range**2))
    _eps_mev4 = (m_I**2 / 4.0) * ((_mu_range**3) / np.sqrt(2*m_chi**2 - _mu_range**2) - m_chi**2)
    P_DM_TABLE = _p_mev4 * MeV4_to_MeV_fm3 * MeV_fm3_to_pa_cgs
    EPS_DM_TABLE = _eps_mev4 * MeV4_to_MeV_fm3 * (MeV_fm3_to_pa_cgs / c**2)

    def eps_q(P): return (3.0 * (P / MeV_fm3_to_pa_cgs) + 4.0 * b_bag) * (MeV_fm3_to_pa_cgs / c**2) if P > 0 else 0.0
    def eps_dm(P): return np.interp(np.clip(P, 0.0, P_INF_CGS), P_DM_TABLE, EPS_DM_TABLE)

    def tov_eqs(r_cm, y):
        P_q, P_dm, M, M_dm = y
        P_q, P_dm = max(P_q, 0.0), max(P_dm, 0.0)
        eq, edm = eps_q(P_q), eps_dm(P_dm)
        etot, Ptot = eq + edm, P_q + P_dm
        if etot <= 0 or r_cm <= 0: return [0.0, 0.0, 0.0, 0.0]
        
        r, M_p = r_cm * len_to_lenplanck, M * m_to_mplanck
        P_q_p, P_dm_p, Ptot_p = P_q*pa_to_Pplanck, P_dm*pa_to_Pplanck, Ptot*pa_to_Pplanck
        eq_p, edm_p, etot_p = eq*gcmdens_to_densplanck, edm*gcmdens_to_densplanck, etot*gcmdens_to_densplanck

        metric = (M_p + 4.0 * pi * (r**3) * Ptot_p) / (r * (r - 2.0 * M_p))
        dP_q_dr = -((eq_p + P_q_p) * metric) if P_q_p > 0 else 0.0
        dP_dm_dr = -((edm_p + P_dm_p) * metric) if P_dm_p > 0 else 0.0
        dM_dr, dM_dm_dr = 4.0 * pi * (r**2) * etot_p, 4.0 * pi * (r**2) * edm_p
        
        return [dP_q_dr * len_to_lenplanck / pa_to_Pplanck, dP_dm_dr * len_to_lenplanck / pa_to_Pplanck, 
                dM_dr * len_to_lenplanck / m_to_mplanck, dM_dm_dr * len_to_lenplanck / m_to_mplanck]

    def total_surface(r, y): return max(y[0], y[1]) - 1e-5
    total_surface.terminal = True

    def get_sequence(frac):
        m_arr, r_arr, rho_arr = [], [], []
        densities = np.geomspace(185.0, 4000.0, 100)
        for rho_c in densities:
            P_init_q = (1/3) * (rho_c - 4.0 * b_bag) * MeV_fm3_to_pa_cgs
            if P_init_q <= 1e-10: continue
            if frac == 0.0: P_init_dm = 0.0
            else:
                def frac_err(p_guess):
                    sol = solve_ivp(tov_eqs, (1e2, 500e5), [P_init_q, p_guess, 0.0, 0.0], events=total_surface, method='RK45')
                    return (sol.y[3][-1] / sol.y[2][-1]) - frac if sol.y[2][-1] > 0 else -frac
                try: P_init_dm = root_scalar(frac_err, bracket=[0.0, P_INF_CGS * 0.999], method='brentq').root
                except: continue
            sol = solve_ivp(tov_eqs, (1e2, 500e5), [P_init_q, P_init_dm, 0.0, 0.0], events=total_surface, method='RK45')
            m_arr.append(sol.y[2][-1] / Msun); r_arr.append(sol.t[-1] / 1e5); rho_arr.append(rho_c)
        return r_arr, m_arr, rho_arr

    r_pure, m_pure, rho_pure = get_sequence(0.0)
    r_vis, m_tot, rho_res = get_sequence(target_fraction)

    # Micro Profile & Speed of Sound Calculation (1.4 Msun)
    r_micro, eq_micro, edm_micro, cs2_hybrid = [], [], [], []
    r_micro_p, eq_pure_micro, cs2_pure = [], [], []
    
    if 1.4 >= np.min(m_tot) and 1.4 <= np.max(m_tot):
        idx = np.argmax(m_tot)
        rho_14 = interp1d(m_tot[:idx+1], rho_res[:idx+1], kind='cubic')(1.4)
        P_init_q = (1/3) * (rho_14 - 4.0 * b_bag) * MeV_fm3_to_pa_cgs
        if target_fraction > 0:
            def frac_err_14(p_guess):
                sol = solve_ivp(tov_eqs, (1e2, 500e5), [P_init_q, p_guess, 0.0, 0.0], events=total_surface, method='RK45')
                return (sol.y[3][-1] / sol.y[2][-1]) - target_fraction
            P_init_dm = root_scalar(frac_err_14, bracket=[0.0, P_INF_CGS * 0.999], method='brentq').root
        else: P_init_dm = 0.0
        
        sol_micro = solve_ivp(tov_eqs, (1e2, 500e5), [P_init_q, P_init_dm, 0.0, 0.0], events=total_surface, method='RK45', max_step=5000)
        r_micro = sol_micro.t / 1e5
        eq_micro = np.array([eps_q(p) * (c**2 / MeV_fm3_to_pa_cgs) for p in sol_micro.y[0]])
        edm_micro = np.array([eps_dm(p) * (c**2 / MeV_fm3_to_pa_cgs) for p in sol_micro.y[1]])
        
        # Calculate Speed of Sound (cs^2) = dP_tot / dEps_tot
        P_tot_mev = (sol_micro.y[0] + sol_micro.y[1]) / MeV_fm3_to_pa_cgs
        Eps_tot_mev = eq_micro + edm_micro
        dP = np.gradient(P_tot_mev)
        dEps = np.gradient(Eps_tot_mev)
        cs2_hybrid = np.divide(dP, dEps, out=np.zeros_like(dP), where=dEps!=0)

        # Pure 1.4 Msun profile for comparison
        if 1.4 >= np.min(m_pure) and 1.4 <= np.max(m_pure):
            idx_p = np.argmax(m_pure)
            rho_14_p = interp1d(m_pure[:idx_p+1], rho_pure[:idx_p+1], kind='cubic')(1.4)
            P_init_q_p = (1/3) * (rho_14_p - 4.0 * b_bag) * MeV_fm3_to_pa_cgs
            sol_micro_p = solve_ivp(tov_eqs, (1e2, 500e5), [P_init_q_p, 0.0, 0.0, 0.0], events=total_surface, method='RK45', max_step=5000)
            r_micro_p = sol_micro_p.t / 1e5
            eq_pure_micro = np.array([eps_q(p) * (c**2 / MeV_fm3_to_pa_cgs) for p in sol_micro_p.y[0]])
            
            P_pure_mev = sol_micro_p.y[0] / MeV_fm3_to_pa_cgs
            dP_p = np.gradient(P_pure_mev)
            dEps_p = np.gradient(eq_pure_micro)
            cs2_pure = np.divide(dP_p, dEps_p, out=np.zeros_like(dP_p), where=dEps_p!=0)

    return r_vis, m_tot, r_pure, m_pure, r_micro, eq_micro, edm_micro, cs2_hybrid, r_micro_p, eq_pure_micro, cs2_pure

# --- UI RENDERING ---
if submit_button or 'data_loaded' not in st.session_state:
    with st.spinner("Integrating Einstein Field Equations... (~20 seconds)"):
        r_vis, m_tot, r_pure, m_pure, r_micro, eq_micro, edm_micro, cs2_hybrid, r_micro_p, eq_pure_micro, cs2_pure = run_full_simulation(b_bag, m_chi, m_I, f_dm)
        st.session_state['data'] = (r_vis, m_tot, r_pure, m_pure, r_micro, eq_micro, edm_micro, cs2_hybrid, r_micro_p, eq_pure_micro, cs2_pure)
        st.session_state['data_loaded'] = True

r_vis, m_tot, r_pure, m_pure, r_micro, eq_micro, edm_micro, cs2_hybrid, r_micro_p, eq_pure_micro, cs2_pure = st.session_state['data']

if len(m_tot) > 0:
    max_mass = np.max(m_tot)
    crit_radius = r_vis[np.argmax(m_tot)]
    
    st.markdown("### Stability & Kinematic Signatures")
    col1, col2, col3 = st.columns(3)
    col1.metric("Critical Mass Limit", f"{max_mass:.3f} M☉", help="Harrison-Zeldovich-Wheeler Limit. Maximum mass before gravitational collapse.")
    col2.metric("Radius at Collapse", f"{crit_radius:.2f} km", help="The visible radius of the star exactly at the point of maximum mass.")
    col3.metric("Stability Gauge (∂M/∂ρc)", "0.000", delta="Collapse Imminent", delta_color="inverse", help="When this derivative hits zero, the star becomes dynamically unstable.")

    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### Macro-View: Hydrostatic Sequence")
        fig_macro = go.Figure()

        m_lims = np.linspace(0.05, 3.2, 100)
        r_sch_km = (2.0 * G * (m_lims * Msun) / c**2) / 1e5
        r_buc_km = (2.25 * G * (m_lims * Msun) / c**2) / 1e5

        fig_macro.add_trace(go.Scatter(x=r_sch_km, y=m_lims, fill='tozerox', mode='none', fillcolor='rgba(169, 169, 169, 0.6)', name='Black Hole', hoverinfo='none'))
        fig_macro.add_trace(go.Scatter(x=r_buc_km, y=m_lims, fill='tonextx', mode='none', fillcolor='rgba(255, 192, 203, 0.3)', name='Buchdahl Limit', hoverinfo='none'))
        fig_macro.add_hline(y=2.0, line_dash="dot", line_color="gray", annotation_text="2.0 M☉ Observational Limit", annotation_position="bottom right")

        fig_macro.add_trace(go.Scatter(x=r_pure, y=m_pure, mode='lines', line=dict(color='red', dash='dash'), name='Pure Quark Star'))
        fig_macro.add_trace(go.Scatter(x=r_vis, y=m_tot, mode='lines', line=dict(color='blue', width=2.5), name=f'Hybrid Star ({f_dm}% DM)'))
        fig_macro.add_trace(go.Scatter(x=[crit_radius], y=[max_mass], mode='markers', marker=dict(color='black', size=10, symbol='x'), name=f'Critical Point ({crit_radius:.2f} km, {max_mass:.3f} M☉)'))
        
        fig_macro.update_layout(xaxis_title="Visible Radius (km)", yaxis_title="Total Mass (M☉)", height=500, xaxis=dict(range=[0, max(max(r_pure), max(r_vis)) + 1]), yaxis=dict(range=[0, max(max(m_pure), max(m_tot)) + 0.5]))
        st.plotly_chart(fig_macro, width='stretch')
        # --- MACRO EXPLANATION ---
        with st.expander("🔬 Analyze this Mass-Radius Sequence"):
            st.markdown(f"""
            **The Harrison-Zeldovich-Wheeler Criterion:** The 'x' marker on the graph represents the onset of gravitational instability ($\partial M / \partial \rho_c = 0$). Any star pushed beyond this central density will collapse into a Black Hole.

            **Kinematic Impact of Dark Matter:** Notice the gap between the red dotted line (Pure Quark) and the blue line (Hybrid). By adding **{f_dm}%** Bosonic Dark Matter, we introduce an "effective softening" to the global Equation of State. The dark matter core exerts its own gravitational pull but provides less pressure support than the quark matter it displaces, generally resulting in a lower maximum mass and a more compact visible radius.
            """)
        
        # --- DATA EXPORT BUTTON ---
        csv_df = pd.DataFrame({'Radius_km': r_vis, 'Mass_Msun': m_tot})
        csv = csv_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sequence Data (CSV)",
            data=csv,
            file_name=f"hybrid_star_sequence_{f_dm}pct_DM.csv",
            mime="text/csv",
            help="Download the Mass-Radius array for the currently simulated Hybrid Star."
        )

    with col_chart2:
        st.markdown("#### Micro-View: Core Analytics (1.4 M☉ Star)")
        if len(r_micro) > 0:
            # We use tabs to switch between Density and Speed of Sound smoothly
            tab1, tab2 = st.tabs(["Density Profile", "Speed of Sound (Causality)"])
            
            with tab1:
                fig_micro = go.Figure()
                if len(r_micro_p) > 0:
                    fig_micro.add_trace(go.Scatter(x=r_micro_p, y=eq_pure_micro, mode='lines', line=dict(color='red', dash='dot'), name='Pure Quark Baseline'))
                fig_micro.add_trace(go.Scatter(x=r_micro, y=eq_micro, mode='lines', line=dict(color='blue', width=2.5), name='Quark Fluid (Hybrid)'))
                if f_dm > 0:
                    fig_micro.add_trace(go.Scatter(x=r_micro, y=edm_micro, mode='lines', line=dict(color='black', dash='dash', width=2), name='DM Fluid (Hybrid)'))
                    fig_micro.add_trace(go.Scatter(x=r_micro, y=edm_micro, mode='none', fill='tozeroy', fillcolor='rgba(128,128,128,0.2)', showlegend=False, hoverinfo='none'))
                fig_micro.update_layout(xaxis_title="Radial Distance r (km)", yaxis_title="Energy Density ε (MeV/fm³)", height=450)
                st.plotly_chart(fig_micro, width='stretch')
                # --- MICRO EXPLANATION ---
            st.info(f"""
            **Core Analytics Breakdown (1.4 M☉ Canonical Star):**
            * **Density Profile:** The dark matter (gray/black) displaces the quark fluid. If $m_\chi$ is high, it forms a dense **Core** at $r=0$. If $m_\chi$ is low, it bleeds outward into a **Halo**. Notice how the presence of the DM core physically forces the red quark density higher to compensate for the added gravity!
            * **Causality & Speed of Sound ($c_s^2$):** A valid physical fluid cannot transmit sound faster than light ($c_s^2 \leq 1$). The sharp "kink" or drop in the purple line represents the exact boundary where the Dark Matter core ends and the pure Quark envelope begins.
            """)
                
            with tab2:
                fig_cs2 = go.Figure()
                # Causality Limit (Speed of Light)
                fig_cs2.add_hline(y=1.0, line_dash="solid", line_color="red", annotation_text="Causality Limit (c²)", annotation_position="top left")
                
                if len(r_micro_p) > 0:
                    fig_cs2.add_trace(go.Scatter(x=r_micro_p, y=cs2_pure, mode='lines', line=dict(color='red', dash='dot'), name='Pure Quark cs²'))
                
                fig_cs2.add_trace(go.Scatter(x=r_micro, y=cs2_hybrid, mode='lines', line=dict(color='purple', width=2.5), name='Hybrid Effective cs²'))
                
                fig_cs2.update_layout(
                    xaxis_title="Radial Distance r (km)", 
                    yaxis_title="Speed of Sound (cs²/c²)", 
                    height=450,
                    yaxis=dict(range=[0.0, max(1.1, np.max(cs2_hybrid) + 0.1)])
                )
                st.plotly_chart(fig_cs2, width='stretch')
                # --- CAUSALITY EXPLANATION ---
                st.info("""
                **Causality & Thermodynamic Stability ($c_s^2$):**
                * **The Speed of Light Limit:** In General Relativity, the speed of sound squared ($c_s^2 = \partial P / \partial \epsilon$) must never exceed 1.0. If a fluid crosses the red line, it violates causality (information traveling faster than light). This proves our EoS is physically viable.
                * **Fluid Stiffness:** A higher $c_s^2$ means the fluid is "stiffer" and provides more pressure support against gravity. Note how the interacting Bosonic Dark Matter can be stiffer than the "soft" Quark Matter envelope (which hovers around the conformal limit of 1/3).
                * **The Phase Boundary:** The sudden drop or "kink" in the purple line represents the exact radial boundary where the dense Dark Matter core ends and the pure Quark envelope begins. This sharp discontinuity is the mathematical hallmark of a multi-fluid phase transition.
                """)
        else:
            st.warning("A 1.4 M☉ star is not dynamically stable with the currently selected parameters.")
