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

# ── Helper ─────────────────────────────────────────────────────────────────────
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
st.markdown("Predict the **environmental impact** of fertiliser application in rice cultivation — without needing a full Life Cycle Assessment.")
st.markdown("---")

# ── Mode Toggle ────────────────────────────────────────────────────────────────
mode = st.radio("Select Mode", ["Single Prediction", "Compare Two Combinations"], horizontal=True)
st.markdown("---")

# ── Ranges Info ────────────────────────────────────────────────────────────────
st.info("**Recommended Ranges (kg/ha):** N: 120–150  |  P: 40–60  |  K: 30–40  |  Zn: 10–30\n\n⚠️ Values outside these ranges may give unreliable predictions.")

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE MODE
# ══════════════════════════════════════════════════════════════════════════════
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

        # ── What's This? ───────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📖 What do these mean?")

        with st.expander("🌍 Global Warming (GWP)"):
            st.markdown("""
            **Global Warming Potential (GWP)** measures how much a process contributes to climate change.
            It's expressed in **kg CO₂-equivalent** — the amount of CO₂ that would have the same warming effect.
            
            In rice farming, GWP is mainly driven by **nitrogen fertiliser** — producing and applying N releases 
            greenhouse gases like N₂O, which is 273x more potent than CO₂.
            """)

        with st.expander("💧 Freshwater Eutrophication"):
            st.markdown("""
            **Freshwater Eutrophication** measures how much a process contributes to excessive nutrient 
            enrichment in freshwater bodies like lakes and rivers.
            
            Too many nutrients (especially **phosphorus**) cause algae to grow out of control, depleting 
            oxygen and killing aquatic life. Expressed in **kg Phosphorus-equivalent**.
            """)

        with st.expander("🌫️ Terrestrial Acidification"):
            st.markdown("""
            **Terrestrial Acidification** measures how much a process contributes to soil and ecosystem 
            acidification through emissions of acidifying substances like ammonia (NH₃) and sulfur dioxide (SO₂).
            
            In rice farming, nitrogen fertiliser releases ammonia which acidifies soil, reducing its 
            fertility over time. Expressed in **kg SO₂-equivalent**.
            """)

        with st.expander("☠️ Terrestrial Ecotoxicity"):
            st.markdown("""
            **Terrestrial Ecotoxicity** measures the toxic impact of chemical substances on land-based 
            ecosystems — soil organisms, plants, and animals.
            
            **Zinc** is the dominant driver here. While Zn is an essential micronutrient for crops, 
            excess application accumulates in soil and becomes toxic to organisms over time. 
            Expressed in **CTUe** (Comparative Toxic Units for ecosystems).
            """)

        st.markdown("---")
        st.caption("Model trained on conventional rice cultivation LCA data using ecoinvent processes. Results are predictions — not a substitute for full LCA.")

# ══════════════════════════════════════════════════════════════════════════════
# COMPARE MODE
# ══════════════════════════════════════════════════════════════════════════════
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

    # Instant validation for both
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

        categories  = ["🌍 Global Warming", "💧 Freshwater Eutrophication", "🌫️ Terrestrial Acidification", "☠️ Terrestrial Ecotoxicity"]
        units       = ["kg CO₂-eq", "kg P-eq", "kg SO₂-eq", "CTUe"]
        values_a    = [gwp_a, eu_a, ac_a, eco_a]
        values_b    = [gwp_b, eu_b, ac_b, eco_b]
        formats     = ["{:,.2f}", "{:.6f}", "{:.4f}", "{:,.2f}"]

        for i, cat in enumerate(categories):
            col1, col2, col3 = st.columns([2, 2, 1])
            a_val = values_a[i]
            b_val = values_b[i]
            fmt   = formats[i]
            winner = "🅰️ Lower" if a_val < b_val else "🅱️ Lower"
            with col1:
                st.metric(f"{cat} — A", f"{fmt.format(a_val)} {units[i]}")
            with col2:
                st.metric(f"{cat} — B", f"{fmt.format(b_val)} {units[i]}")
            with col3:
                st.markdown(f"<br><b>{winner}</b>", unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Lower values = less environmental impact. Model trained on conventional rice cultivation LCA data.")
        # ── Model Validation ───────────────────────────────────────────────────────────
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