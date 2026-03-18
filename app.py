import streamlit as st
import pandas as pd
import joblib

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rice Fertiliser Impact Predictor",
    page_icon="🌱",
    layout="centered"
)

# ── Load Models ────────────────────────────────────────────────────────────────
model1 = joblib.load("model1_env_impacts.pkl")
model2 = joblib.load("model2_ecotoxicity.pkl")

# ── Helper Functions ───────────────────────────────────────────────────────────
def predict(N, P, K, Zn):
    env_inputs = pd.DataFrame([[N, P, K, Zn]], columns=['N_rate', 'P_rate', 'K_rate', 'Zn_rate'])
    eco_inputs = pd.DataFrame([[Zn]], columns=['Zn_rate'])
    env_pred = model1.predict(env_inputs)[0]
    eco_pred = model2.predict(eco_inputs)[0]
    return env_pred[0], env_pred[1], env_pred[2], eco_pred

def validate(N, P, K, Zn):
    ranges = {'N': (120, 150), 'P': (40, 60), 'K': (30, 40), 'Zn': (10, 30)}
    inputs = {'N': N, 'P': P, 'K': K, 'Zn': Zn}
    return [f"**{k}** = {v} kg (valid: {ranges[k][0]}–{ranges[k][1]} kg)"
            for k, v in inputs.items() if not (ranges[k][0] <= v <= ranges[k][1])]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🌱 Rice Fertiliser Impact Predictor")
st.markdown("A tool for predicting the **environmental impact** of fertiliser application in rice cultivation — without needing a full Life Cycle Assessment.")
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔬 LCA Impact Predictor", "🌾 Field Emission Calculator"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LCA IMPACT PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.subheader("Predict Environmental Impact Scores")
    st.markdown("Uses a trained ML model to predict LCA impact scores from fertiliser inputs.")

    mode = st.radio("Select Mode", ["Single Prediction", "Compare Two Combinations"], horizontal=True)
    st.markdown("---")

    st.info("**Recommended Ranges (kg/ha):** N: 120–150  |  P: 40–60  |  K: 30–40  |  Zn: 10–30\n\n⚠️ Values outside these ranges may give unreliable predictions.")

    # ── Single Mode ────────────────────────────────────────────────────────────
    if mode == "Single Prediction":
        st.subheader("Enter Fertiliser Application Rates (kg/ha)")
        col1, col2 = st.columns(2)
        with col1:
            N  = st.number_input("Nitrogen (N)",   min_value=0.0, value=135.0, step=0.5)
            K  = st.number_input("Potassium (K)",  min_value=0.0, value=35.0,  step=0.5)
        with col2:
            P  = st.number_input("Phosphorus (P)", min_value=0.0, value=50.0,  step=0.5)
            Zn = st.number_input("Zinc (Zn)",      min_value=0.0, value=20.0,  step=0.5)

        out_of_range = validate(N, P, K, Zn)
        if out_of_range:
            st.error("🚨 Out of range:\n\n" + "\n".join(f"- {w}" for w in out_of_range))
        else:
            st.success("✅ All inputs within recommended ranges!")

        st.markdown("---")
        if st.button("🔍 Predict Environmental Impact", use_container_width=True):
            gwp, eu, ac, eco = predict(N, P, K, Zn)

            st.subheader("📊 Predicted Environmental Impact Scores")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🌍 Global Warming",            f"{gwp:,.2f} kg CO₂-eq")
                st.metric("💧 Freshwater Eutrophication", f"{eu:.6f} kg P-eq")
            with col2:
                st.metric("🌫️ Terrestrial Acidification", f"{ac:.4f} kg SO₂-eq")
                st.metric("☠️ Terrestrial Ecotoxicity",   f"{eco:,.2f} CTUe")

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

            with st.expander("☠️ Terrestrial Ecotoxicity"):
                st.markdown("""
                **Terrestrial Ecotoxicity** measures the toxic impact on land-based ecosystems.
                **Zinc** is the dominant driver — excess Zn accumulates in soil and becomes toxic to organisms.
                Expressed in **CTUe** (Comparative Toxic Units for ecosystems).
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
            Zn_a = st.number_input("Zinc (Zn) — A",      min_value=0.0, value=10.0,  step=0.5, key="Zna")

        with col_b:
            st.markdown("### 🅱️ Combination B")
            N_b  = st.number_input("Nitrogen (N) — B",   min_value=0.0, value=150.0, step=0.5, key="Nb")
            P_b  = st.number_input("Phosphorus (P) — B", min_value=0.0, value=60.0,  step=0.5, key="Pb")
            K_b  = st.number_input("Potassium (K) — B",  min_value=0.0, value=40.0,  step=0.5, key="Kb")
            Zn_b = st.number_input("Zinc (Zn) — B",      min_value=0.0, value=30.0,  step=0.5, key="Znb")

        warn_a = validate(N_a, P_a, K_a, Zn_a)
        warn_b = validate(N_b, P_b, K_b, Zn_b)
        if warn_a:
            st.error("🚨 Combination A out of range:\n\n" + "\n".join(f"- {w}" for w in warn_a))
        if warn_b:
            st.error("🚨 Combination B out of range:\n\n" + "\n".join(f"- {w}" for w in warn_b))

        st.markdown("---")
        if st.button("🔍 Compare Combinations", use_container_width=True):
            gwp_a, eu_a, ac_a, eco_a = predict(N_a, P_a, K_a, Zn_a)
            gwp_b, eu_b, ac_b, eco_b = predict(N_b, P_b, K_b, Zn_b)

            st.subheader("📊 Comparison Results")
            categories = ["🌍 Global Warming", "💧 Freshwater Eutrophication",
                         "🌫️ Terrestrial Acidification", "☠️ Terrestrial Ecotoxicity"]
            units      = ["kg CO₂-eq", "kg P-eq", "kg SO₂-eq", "CTUe"]
            values_a   = [gwp_a, eu_a, ac_a, eco_a]
            values_b   = [gwp_b, eu_b, ac_b, eco_b]
            formats    = ["{:,.2f}", "{:.6f}", "{:.4f}", "{:,.2f}"]

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

    # ── Model Validation ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Model Validation")
    st.markdown("The following plots validate the ML models trained on rice cultivation LCA data.")

    with st.expander("📉 Predicted vs Actual"):
        st.image("Plots/plot1_predicted_vs_actual.png", use_container_width=True)
        st.caption("Each point represents a test sample. Points close to the red line indicate accurate predictions.")

    with st.expander("📊 Feature Importance"):
        st.image("Plots/plot2_feature_importance.png", use_container_width=True)
        st.caption("Shows how much each fertiliser input (N, P, K, Zn) influences each impact category.")

    with st.expander("📈 Input vs Impact"):
        st.image("Plots/plot3_input_vs_impact.png", use_container_width=True)
        st.caption("Shows how each impact score changes as individual fertiliser inputs increase.")

    # ── Fertiliser Impact Intensity ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚗️ Fertiliser Impact Intensity")
    st.markdown("Environmental impact caused by **1 kg of each fertiliser input**, calculated independently in OpenLCA.")

    impact_data = {
        "Input": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Zinc (Zn)"],
        "Global Warming (kg CO₂-eq)": [4.964134, 2.906595, 3.016340, 0.777299],
        "Terrestrial Acidification (kg SO₂-eq)": [0.021555, 0.014359, 0.012404, 0.006452],
        "Freshwater Eutrophication (kg P-eq)": [0.001469, 0.001000, 0.000679, 0.000460],
        "Terrestrial Ecotoxicity (CTUe)": [5.186745, 4.168297, 2.671163, 612.915864],
    }
    df_intensity = pd.DataFrame(impact_data)
    st.dataframe(df_intensity, use_container_width=True, hide_index=True)
    st.info("💡 **Zinc (Zn)** has dramatically higher ecotoxicity per kg (612.9 CTUe) compared to N, P and K (2.7–5.2 CTUe). This is why Zn is the sole driver of Terrestrial Ecotoxicity in the ML model.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FIELD EMISSION CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.subheader("🌾 Field Emission Calculator")
    st.markdown("Calculate direct field emissions from fertiliser application based on **IPCC** and **SALCA** methodologies.")
    st.markdown("---")

    # ── Inputs ─────────────────────────────────────────────────────────────────
    st.subheader("Inputs")

    col1, col2 = st.columns(2)
    with col1:
        irrigation = st.selectbox("Irrigation Mode", ["Fully Flooded", "AWD (Alternate Wetting & Drying)"])
        FSN = st.number_input("Synthetic Nitrogen — FSN (kg/ha)", min_value=0.0, value=135.0, step=0.5)
        P_syn = st.number_input("Synthetic Phosphorus — P (kg/ha)", min_value=0.0, value=50.0, step=0.5)

    with col2:
        amendment = st.selectbox("Organic Amendment", ["None", "FYM", "Vermicompost"])
        if amendment != "None":
            amendment_kg = st.number_input(f"{amendment} amount (kg/ha)", min_value=0.0, value=1000.0, step=10.0)
        else:
            amendment_kg = 0.0

    # ── Derived Values ─────────────────────────────────────────────────────────
    amendment_composition = {
        "None":         {"N": 0.0,  "P2O5": 0.0,  "K2O": 0.0},
        "FYM":          {"N": 0.5,  "P2O5": 0.2,  "K2O": 0.5},
        "Vermicompost": {"N": 1.5,  "P2O5": 0.9,  "K2O": 1.2},
    }

    comp      = amendment_composition[amendment]
    FON       = amendment_kg * comp["N"] / 100
    P_organic = amendment_kg * comp["P2O5"] / 100 * 0.436
    N_total   = FSN + FON
    P_total   = P_syn + P_organic
    SFo       = 1.4 if amendment != "None" else 1.0
    flooded   = irrigation == "Fully Flooded"
    SFw       = 1.0 if flooded else 0.55

    # ── Show derived values ────────────────────────────────────────────────────
    if amendment != "None":
        st.info(f"""
        **Derived from {amendment} ({amendment_kg} kg/ha):**
        FON = {FON:.2f} kg N/ha  |  P from organic = {P_organic:.2f} kg P/ha  |  
        Total N = {N_total:.2f} kg/ha  |  Total P = {P_total:.2f} kg/ha
        """)

    st.markdown("---")

    # ── Calculate Button ───────────────────────────────────────────────────────
    if st.button("🧮 Calculate Field Emissions", use_container_width=True):

        # CH₄
        ch4 = 107.1 * SFo * SFw

        # N₂O
        EF_n2o  = 0.003 if flooded else 0.0073
        n2o_n   = N_total * EF_n2o
        n2o     = n2o_n * (44 / 28)

        # NO₃
        LF      = 0.009 if flooded else 0.018
        no3_n   = N_total * LF
        no3     = no3_n * (62 / 14)

        # NH₃
        EF_nh3  = 0.30 if flooded else 0.15
        nh3_n   = N_total * EF_nh3
        nh3     = nh3_n * (17 / 14)

        # PO₄
        RF      = 0.01 if flooded else 0.0055
        p_runoff = P_total * RF
        po4     = p_runoff * (95 / 31)

        # ── Results ────────────────────────────────────────────────────────────
        st.subheader("📊 Field Emission Results (kg/ha/season)")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔥 Methane (CH₄)",         f"{ch4:.4f} kg/ha/season",    help="IPCC methodology")
            st.metric("⚡ Nitrous Oxide (N₂O)",    f"{n2o:.4f} kg/ha/season",    help="IPCC methodology")
            st.metric("💧 Nitrate (NO₃)",          f"{no3:.4f} kg/ha/season",    help="IPCC methodology")
        with col2:
            st.metric("💨 Ammonia (NH₃)",          f"{nh3:.4f} kg/ha/season",    help="IPCC methodology")
            st.metric("🌊 Phosphate (PO₄)",        f"{po4:.4f} kg/ha/season",    help="SALCA methodology")

        st.markdown("---")

        # ── Summary Table ──────────────────────────────────────────────────────
        st.subheader("📋 Summary")
        results_df = pd.DataFrame({
            "Emission": ["CH₄", "N₂O", "NO₃", "NH₃", "PO₄"],
            "Value (kg/ha/season)": [ch4, n2o, no3, nh3, po4],
            "Methodology": ["IPCC", "IPCC", "IPCC", "IPCC", "SALCA"]
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.caption("Calculations based on IPCC (2006) Guidelines for National Greenhouse Gas Inventories and SALCA methodology. Results are estimates for conventional rice cultivation.")