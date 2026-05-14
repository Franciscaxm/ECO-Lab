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
st.markdown("### Interactive Two-Fluid TOV Solver")

# --- PHYSICS GUIDE ---
with st.expander("📖 Physics Guide: How to use this laboratory"):
    st.markdown(r"""
    This simulator integrates the **Tolman-Oppenheimer-Volkoff (TOV)** equations for a two-fluid hybrid star:
    
    $$\frac{dP}{dr} = -\frac{G}{c^2} \frac{(M + 4\pi r^3 P/c^2)(\epsilon + P)}{r(r - 2GM/c^2)}$$
    
    **Parameter Implications:**
    * **Bag Constant ($B$):** Controls Quark Matter confinement. Higher $B$ $\rightarrow$ softer EoS $\rightarrow$ smaller stars.
    * **Boson Mass ($m_\chi$):** Determines DM distribution. High mass $\rightarrow$ compact **Core**. Low mass $\rightarrow$ extended **Halo**.
    * **Interaction Scale ($m_I$):** Controls DM repulsion. Higher $m_I$ $\rightarrow$ stiffer DM $\rightarrow$ higher mass support.
    * **DM Fraction:** Total percentage of Dark Matter mass.
    """)

# --- PHYSICS CONSTANTS (CGS) ---
pi = math.pi
G = 6.67430e-8; c = 2.99792458e10; hbar = 1.054571817e-27
Msun = 1.98840987e33; MeV_fm3_to_pa_cgs = 1.602176634e33  
hbar_c_mev_fm = 197.3269804; MeV4_to_MeV_fm3 = 1.0 / (hbar_c_mev_fm**3) 
m_planck_cgs = math.sqrt((hbar * c) / G)      
l_planck_cgs = math.sqrt((hbar * G) / c**3)   
t_planck_cgs = math.sqrt((hbar * G) / c**5)   
m_to_mplanck, len_to_lenplanck, t_to_tplanck = 1.0/m_planck_cgs, 1.0/l_planck_cgs, 1.0/t_planck_cgs
gcmdens_to_densplanck = m_to_mplanck / (len_to_lenplanck**3)
pa_to_Pplanck = m_to_mplanck / (len_to_lenplanck * (t_to_tplanck**2))

# --- SIDEBAR: PARAMETERS ---
st.sidebar.header("Fundamental Parameters")
with st.sidebar.form("physics_params"):
    st.markdown("**Quark Matter (MIT Bag)**")
    b_bag = st.slider("Bag Constant (B) [MeV/fm³]", 40.0, 60.0, 45.0, 1.0)
    
    st.markdown("**Dark Matter (Bosonic BEC)**")
    m_chi = st.slider("Boson Mass (m_χ) [MeV]", 5.0, 300.0, 100.0, 5.0)
    m_I = st.slider("Interaction Scale (m_I) [MeV]", 50.0, 500.0, 250.0, 10.0)
    
    st.markdown("**Stellar Composition**")
    f_dm = st.slider("Dark Matter Fraction (%)", 0.0, 20.0, 5.0, 1.0)
    
    submit_button = st.form_submit_button(label="🚀 Run TOV Simulation")

# --- CORE PHYSICS ENGINE ---
@st.cache_data(show_spinner=False)
def run_full_simulation(b_bag, m_chi, m_I, f_dm):
    target_fraction = f_dm / 100.0
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
        Ptot_p, etot_p = Ptot*pa_to_Pplanck, etot*gcmdens_to_densplanck
        metric = (M_p + 4.0 * pi * (r**3) * Ptot_p) / (r * (r - 2.0 * M_p))
        
        dP_q_dr = -((eps_q(P_q)*gcmdens_to_densplanck + P_q*pa_to_Pplanck) * metric) if P_q > 0 else 0.0
        dP_dm_dr = -((eps_dm(P_dm)*gcmdens_to_densplanck + P_dm*pa_to_Pplanck) * metric) if P_dm > 0 else 0.0
        dM_dr, dM_dm_dr = 4.0 * pi * (r**2) * etot_p, 4.0 * pi * (r**2) * (edm*gcmdens_to_densplanck)
        
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
        
        dP = np.gradient((sol_micro.y[0] + sol_micro.y[1]) / MeV_fm3_to_pa_cgs)
        dEps = np.gradient(eq_micro + edm_micro)
        cs2_hybrid = np.divide(dP, dEps, out=np.zeros_like(dP), where=dEps!=0)

        if 1.4 >= np.min(m_pure) and 1.4 <= np.max(m_pure):
            idx_p = np.argmax(m_pure)
            rho_14_p = interp1d(m_pure[:idx_p+1], rho_pure[:idx_p+1], kind='cubic')(1.4)
            P_init_q_p = (1/3) * (rho_14_p - 4.0 * b_bag) * MeV_fm3_to_pa_cgs
            sol_micro_p = solve_ivp(tov_eqs, (1e2, 500e5), [P_init_q_p, 0.0, 0.0, 0.0], events=total_surface, method='RK45', max_step=5000)
            r_micro_p = sol_micro_p.t / 1e5
            eq_pure_micro = np.array([eps_q(p) * (c**2 / MeV_fm3_to_pa_cgs) for p in sol_micro_p.y[0]])
            cs2_pure = np.divide(np.gradient(sol_micro_p.y[0] / MeV_fm3_to_pa_cgs), np.gradient(eq_pure_micro), out=np.zeros_like(r_micro_p), where=np.gradient(eq_pure_micro)!=0)

    return r_vis, m_tot, r_pure, m_pure, r_micro, eq_micro, edm_micro, cs2_hybrid, r_micro_p, eq_pure_micro, cs2_pure

# --- EXECUTION ---
if submit_button or 'data_loaded' not in st.session_state:
    with st.spinner("Integrating Einstein Field Equations..."):
        results = run_full_simulation(b_bag, m_chi, m_I, f_dm)
        st.session_state['data'] = results
        st.session_state['data_loaded'] = True

r_vis, m_tot, r_pure, m_pure, r_micro, eq_micro, edm_micro, cs2_hybrid, r_micro_p, eq_pure_micro, cs2_pure = st.session_state['data']

# --- SHARED PLOTLY LAYOUT (MOBILE OPTIMIZED) ---
def get_layout(title, xtitle, ytitle):
    return dict(
        title=title,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=14),
        xaxis=dict(title=xtitle, title_font=dict(size=16), tickfont=dict(size=12)),
        yaxis=dict(title=ytitle, title_font=dict(size=16), tickfont=dict(size=12)),
        margin=dict(l=10, r=10, t=50, b=10),
        height=480,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

# --- RENDERING ---
if len(m_tot) > 0:
    max_mass, crit_radius = np.max(m_tot), r_vis[np.argmax(m_tot)]
    col1, col2, col3 = st.columns(3)
    col1.metric("Critical Mass Limit", f"{max_mass:.3f} M☉")
    col2.metric("Radius at Collapse", f"{crit_radius:.2f} km")
    col3.metric("Stability Gauge", "0.000", delta="Stable" if max_mass > 2.0 else "Critical", delta_color="normal")

    c_macro, c_micro = st.columns(2)
    
    with c_macro:
        fig_macro = go.Figure()
        m_lims = np.linspace(0.05, 3.5, 100)
        r_sch_km = (2.0 * G * (m_lims * Msun) / c**2) / 1e5
        r_buc_km = (2.25 * G * (m_lims * Msun) / c**2) / 1e5
        fig_macro.add_trace(go.Scatter(x=r_sch_km, y=m_lims, fill='tozerox', mode='none', fillcolor='rgba(169, 169, 169, 0.4)', name='Black Hole'))
        fig_macro.add_trace(go.Scatter(x=r_buc_km, y=m_lims, fill='tonextx', mode='none', fillcolor='rgba(255, 192, 203, 0.2)', name='Buchdahl Limit'))
        fig_macro.add_trace(go.Scatter(x=r_pure, y=m_pure, mode='lines', line=dict(color='red', dash='dash'), name='Pure Quark'))
        fig_macro.add_trace(go.Scatter(x=r_vis, y=m_tot, mode='lines', line=dict(color='#4b7bff', width=3), name=f'Hybrid ({f_dm}%)'))
        fig_macro.update_layout(get_layout("Macro: M-R Sequence", "Radius (km)", "Mass (M☉)"))
        st.plotly_chart(fig_macro, width='stretch')
        
        with st.expander("🔬 Analyze M-R Graph"):
            st.markdown(f"Adding **{f_dm}%** DM causes an effective softening of the EoS, shifting the sequence toward the Schwarzschild limit.")
        
        csv = pd.DataFrame({'Radius_km': r_vis, 'Mass_Msun': m_tot}).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Data (CSV)", csv, f"ECO_Data_{f_dm}pct.csv", "text/csv")

    with c_micro:
        if len(r_micro) > 0:
            tab1, tab2 = st.tabs(["Density Profile", "Causality (cs²)"])
            with tab1:
                fig_d = go.Figure()
                if len(r_micro_p) > 0: fig_d.add_trace(go.Scatter(x=r_micro_p, y=eq_pure_micro, mode='lines', line=dict(color='red', dash='dot'), name='Pure Quark'))
                fig_d.add_trace(go.Scatter(x=r_micro, y=eq_micro, mode='lines', line=dict(color='#4b7bff', width=2), name='Quark Fluid'))
                if f_dm > 0: fig_d.add_trace(go.Scatter(x=r_micro, y=edm_micro, mode='lines', fill='tozeroy', line=dict(color='white', dash='dash'), name='DM Fluid'))
                fig_d.update_layout(get_layout("Micro: Density", "r (km)", "ε (MeV/fm³)"))
                st.plotly_chart(fig_d, width='stretch')
            with tab2:
                fig_cs = go.Figure()
                fig_cs.add_hline(y=1.0, line_dash="solid", line_color="red", annotation_text="Causality Limit")
                fig_cs.add_trace(go.Scatter(x=r_micro, y=cs2_hybrid, mode='lines', line=dict(color='#b04bff', width=2), name='Hybrid cs²'))
                fig_cs.update_layout(get_layout("Micro: Sound Speed", "r (km)", "cs²/c²"))
                st.plotly_chart(fig_cs, width='stretch')
            
            st.info("The sudden drop in Sound Speed represents the boundary of the Dark Matter core.")
