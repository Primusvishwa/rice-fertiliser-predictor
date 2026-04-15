import streamlit as st
import pandas as pd
import joblib

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rice Fertiliser Impact Predictor",
    page_icon="🌱",
    layout="centered"
)

# ── Load Model ─────────────────────────────────────────────────────────────────
model = joblib.load("model_all_impacts.pkl")

# ── Helper Functions ───────────────────────────────────────────────────────────
def predict(N, P, K):
    inputs = pd.DataFrame([[N, P, K]], columns=['N_rate', 'P_rate', 'K_rate'])
    pred = model.predict(inputs)[0]
    return pred[0], pred[1], pred[2]

def validate(N, P, K):
    ranges = {'N': (120, 150), 'P': (40, 60), 'K': (30, 40)}
    inputs = {'N': N, 'P': P, 'K': K}
    return [f"**{k}** = {v} kg (valid: {ranges[k][0]}–{ranges[k][1]} kg)"
            for k, v in inputs.items() if not (ranges[k][0] <= v <= ranges[k][1])]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🌱 Rice Fertiliser Impact Predictor")
st.markdown("Predict the **environmental impact** of fertiliser application in rice cultivation — without needing a full Life Cycle Assessment.")
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔬 LCA Impact Predictor", "🌾 Field Emission Calculator"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LCA IMPACT PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.subheader("Predict Environmental Impact Scores")
    st.markdown("Uses a trained **Multivariate Linear Regression** model to predict LCA impact scores from fertiliser inputs.")

    mode = st.radio("Select Mode", ["Single Prediction", "Compare Two Combinations"], horizontal=True)
    st.markdown("---")

    st.info("**Recommended Ranges (kg/ha):** N: 120–150  |  P: 40–60  |  K: 30–40\n\n⚠️ Values outside these ranges may give unreliable predictions.")

    # ── Single Mode ────────────────────────────────────────────────────────────
    if mode == "Single Prediction":
        st.subheader("Enter Fertiliser Application Rates (kg/ha)")
        col1, col2 = st.columns(2)
        with col1:
            N  = st.number_input("Nitrogen (N)",   min_value=0.0, value=135.0, step=0.5)
        with col2:
            P  = st.number_input("Phosphorus (P)", min_value=0.0, value=50.0,  step=0.5)
        K  = st.number_input("Potassium (K)",  min_value=0.0, value=35.0,  step=0.5)

        out_of_range = validate(N, P, K)
        if out_of_range:
            st.error("🚨 Out of range:\n\n" + "\n".join(f"- {w}" for w in out_of_range))
        else:
            st.success("✅ All inputs within recommended ranges!")

        st.markdown("---")
        if st.button("🔍 Predict Environmental Impact", use_container_width=True):
            gwp, eu, ac = predict(N, P, K)

            st.subheader("📊 Predicted Environmental Impact Scores")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🌍 Global Warming",            f"{gwp:,.2f} kg CO₂-eq")
                st.metric("💧 Freshwater Eutrophication", f"{eu:.6f} kg P-eq")
            with col2:
                st.metric("🌫️ Terrestrial Acidification", f"{ac:.4f} kg SO₂-eq")

            st.markdown("---")
            st.subheader("📖 What do these mean?")

            with st.expander("🌍 Global Warming (GWP)"):
                st.markdown("""
                **Global Warming Potential (GWP)** measures how much a process contributes to climate change.
                Expressed in **kg CO₂-equivalent**.
                In rice farming, GWP is mainly driven by **nitrogen fertiliser** — producing and applying N releases
                greenhouse gases like N₂O, which is 273x more potent than CO₂.
                """)

            with st.expander("💧 Freshwater Eutrophication"):
                st.markdown("""
                **Freshwater Eutrophication** measures nutrient enrichment in freshwater bodies.
                Too many nutrients (especially **phosphorus**) cause algae blooms, depleting oxygen and killing aquatic life.
                Expressed in **kg Phosphorus-equivalent**.
                """)

            with st.expander("🌫️ Terrestrial Acidification"):
                st.markdown("""
                **Terrestrial Acidification** measures soil and ecosystem acidification through emissions of
                ammonia (NH₃) and sulfur dioxide (SO₂).
                Nitrogen fertiliser releases ammonia which acidifies soil over time.
                Expressed in **kg SO₂-equivalent**.
                """)

            st.markdown("---")
            st.caption("Model trained on conventional rice cultivation LCA data using ecoinvent processes. Results are predictions — not a substitute for full LCA.")

    # ── Compare Mode ───────────────────────────────────────────────────────────
    else:
        st.subheader("Compare Two Fertiliser Combinations")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### 🅰️ Combination A")
            N_a  = st.number_input("Nitrogen (N) — A",   min_value=0.0, value=120.0, step=0.5, key="Na")
            P_a  = st.number_input("Phosphorus (P) — A", min_value=0.0, value=40.0,  step=0.5, key="Pa")
            K_a  = st.number_input("Potassium (K) — A",  min_value=0.0, value=30.0,  step=0.5, key="Ka")

        with col_b:
            st.markdown("### 🅱️ Combination B")
            N_b  = st.number_input("Nitrogen (N) — B",   min_value=0.0, value=150.0, step=0.5, key="Nb")
            P_b  = st.number_input("Phosphorus (P) — B", min_value=0.0, value=60.0,  step=0.5, key="Pb")
            K_b  = st.number_input("Potassium (K) — B",  min_value=0.0, value=40.0,  step=0.5, key="Kb")

        warn_a = validate(N_a, P_a, K_a)
        warn_b = validate(N_b, P_b, K_b)
        if warn_a:
            st.error("🚨 Combination A out of range:\n\n" + "\n".join(f"- {w}" for w in warn_a))
        if warn_b:
            st.error("🚨 Combination B out of range:\n\n" + "\n".join(f"- {w}" for w in warn_b))

        st.markdown("---")
        if st.button("🔍 Compare Combinations", use_container_width=True):
            gwp_a, eu_a, ac_a = predict(N_a, P_a, K_a)
            gwp_b, eu_b, ac_b = predict(N_b, P_b, K_b)

            st.subheader("📊 Comparison Results")
            categories = ["🌍 Global Warming", "💧 Freshwater Eutrophication",
                         "🌫️ Terrestrial Acidification"]
            units      = ["kg CO₂-eq", "kg P-eq", "kg SO₂-eq"]
            values_a   = [gwp_a, eu_a, ac_a]
            values_b   = [gwp_b, eu_b, ac_b]
            formats    = ["{:,.2f}", "{:.6f}", "{:.4f}"]

            for i, cat in enumerate(categories):
                col1, col2, col3 = st.columns([2, 2, 1])
                fmt    = formats[i]
                winner = "🅰️ Lower" if values_a[i] < values_b[i] else "🅱️ Lower"
                with col1:
                    st.metric(f"{cat} — A", f"{fmt.format(values_a[i])} {units[i]}")
                with col2:
                    st.metric(f"{cat} — B", f"{fmt.format(values_b[i])} {units[i]}")
                with col3:
                    st.markdown(f"<br><b>{winner}</b>", unsafe_allow_html=True)

            st.markdown("---")
            st.caption("Lower values = less environmental impact.")

    # ── Fertiliser Impact Intensity ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚗️ Fertiliser Impact Intensity")
    st.markdown("Environmental impact caused by **1 kg of each fertiliser input**, calculated independently in OpenLCA.")

    impact_data = {
        "Input": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"],
        "Global Warming (kg CO₂-eq)": [4.964134, 2.906595, 3.016340],
        "Terrestrial Acidification (kg SO₂-eq)": [0.021555, 0.014359, 0.012404],
        "Freshwater Eutrophication (kg P-eq)": [0.001469, 0.001000, 0.000679],
    }
    df_intensity = pd.DataFrame(impact_data)
    st.dataframe(df_intensity, use_container_width=True, hide_index=True)
    st.info("💡 **Nitrogen (N)** has the highest Global Warming impact per kg (4.96 kg CO₂-eq) due to N₂O emissions during application.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FIELD EMISSION CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.subheader("🌾 Field Emission Calculator")
    st.markdown("Calculate direct field emissions from fertiliser application based on **IPCC** and **SALCA** methodologies.")
    st.markdown("---")

    st.subheader("Inputs")

    col1, col2 = st.columns(2)
    with col1:
        irrigation = st.selectbox("Irrigation Mode", ["Fully Flooded", "AWD (Alternate Wetting & Drying)"])
        FSN   = st.number_input("Synthetic Nitrogen — FSN (kg/ha)",  min_value=0.0, value=135.0, step=0.5)
        P_syn = st.number_input("Synthetic Phosphorus — P (kg/ha)",  min_value=0.0, value=50.0,  step=0.5)

    with col2:
        amendment1 = st.selectbox("Organic Amendment 1", ["None", "Farm-Yard Manure", "Vermicompost"], key="am1")
        if amendment1 != "None":
            amt1 = st.number_input(f"{amendment1} amount (kg/ha)", min_value=0.0, value=1000.0, step=10.0, key="amt1")
        else:
            amt1 = 0.0

        amendment2 = st.selectbox("Organic Amendment 2", ["None", "Farm-Yard Manure", "Vermicompost"], key="am2")
        if amendment2 != "None":
            amt2 = st.number_input(f"{amendment2} amount (kg/ha)", min_value=0.0, value=1000.0, step=10.0, key="amt2")
        else:
            amt2 = 0.0

    amendment_composition = {
        "None":             {"N": 0.0, "P2O5": 0.0, "K2O": 0.0},
        "Farm-Yard Manure": {"N": 0.5, "P2O5": 0.2, "K2O": 0.5},
        "Vermicompost":     {"N": 1.5, "P2O5": 0.9, "K2O": 1.2},
    }

    comp1     = amendment_composition[amendment1]
    comp2     = amendment_composition[amendment2]
    FON       = (amt1 * comp1["N"] / 100) + (amt2 * comp2["N"] / 100)
    P_organic = (amt1 * comp1["P2O5"] / 100 * 0.436) + (amt2 * comp2["P2O5"] / 100 * 0.436)
    N_total   = FSN + FON
    P_total   = P_syn + P_organic
    SFo       = 1.4 if (amendment1 != "None" or amendment2 != "None") else 1.0
    flooded   = irrigation == "Fully Flooded"
    SFw       = 1.0 if flooded else 0.55

    if amendment1 != "None" or amendment2 != "None":
        lines = []
        if amendment1 != "None":
            lines.append(f"**{amendment1}** ({amt1} kg/ha) → FON = {amt1 * comp1['N'] / 100:.2f} kg N/ha | P = {amt1 * comp1['P2O5'] / 100 * 0.436:.2f} kg P/ha")
        if amendment2 != "None":
            lines.append(f"**{amendment2}** ({amt2} kg/ha) → FON = {amt2 * comp2['N'] / 100:.2f} kg N/ha | P = {amt2 * comp2['P2O5'] / 100 * 0.436:.2f} kg P/ha")
        lines.append(f"**Total N = {N_total:.2f} kg/ha  |  Total P = {P_total:.2f} kg/ha**")
        st.info("\n\n".join(lines))

    st.markdown("---")

    if st.button("🧮 Calculate Field Emissions", use_container_width=True):

        ch4      = 107.1 * SFo * SFw
        EF_n2o   = 0.003 if flooded else 0.0073
        n2o      = N_total * EF_n2o * (44 / 28)
        no3      = N_total * (0.009 if flooded else 0.018) * (62 / 14)
        nh3      = N_total * (0.30 if flooded else 0.15) * (17 / 14)
        po4      = P_total * (0.01 if flooded else 0.0055) * (95 / 31)

        st.subheader("📊 Field Emission Results (kg/ha/season)")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔥 Methane (CH₄)",       f"{ch4:.4f} kg/ha/season", help="IPCC methodology")
            st.metric("⚡ Nitrous Oxide (N₂O)",  f"{n2o:.4f} kg/ha/season", help="IPCC methodology")
            st.metric("💧 Nitrate (NO₃)",        f"{no3:.4f} kg/ha/season", help="IPCC methodology")
        with col2:
            st.metric("💨 Ammonia (NH₃)",        f"{nh3:.4f} kg/ha/season", help="IPCC methodology")
            st.metric("🌊 Phosphate (PO₄)",      f"{po4:.4f} kg/ha/season", help="SALCA methodology")

        st.markdown("---")
        st.subheader("📋 Summary")
        results_df = pd.DataFrame({
            "Emission":             ["CH₄", "N₂O", "NO₃", "NH₃", "PO₄"],
            "Value (kg/ha/season)": [round(ch4,4), round(n2o,4), round(no3,4), round(nh3,4), round(po4,4)],
            "Methodology":          ["IPCC", "IPCC", "IPCC", "IPCC", "SALCA"]
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.caption("Calculations based on IPCC (2006) Guidelines for National Greenhouse Gas Inventories and SALCA methodology. Results are estimates for conventional rice cultivation.")