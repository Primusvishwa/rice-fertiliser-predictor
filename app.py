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

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🌱 Rice Fertiliser Impact Predictor")
st.markdown("Predict the **environmental impact** of fertiliser application in rice cultivation — without needing a full Life Cycle Assessment.")
st.markdown("---")

# ── Input Section ──────────────────────────────────────────────────────────────
st.subheader("Enter Fertiliser Application Rates (kg/ha)")

col1, col2 = st.columns(2)

with col1:
    N  = st.number_input("Nitrogen (N)",    min_value=0.0, value=135.0, step=0.5)
    K  = st.number_input("Potassium (K)",   min_value=0.0, value=35.0,  step=0.5)

with col2:
    P  = st.number_input("Phosphorus (P)",  min_value=0.0, value=50.0,  step=0.5)
    Zn = st.number_input("Zinc (Zn)",       min_value=0.0, value=20.0,  step=0.5)

# ── Validation ─────────────────────────────────────────────────────────────────
ranges = {'N': (120, 150), 'P': (40, 60), 'K': (30, 40), 'Zn': (10, 30)}
inputs = {'N': N, 'P': P, 'K': K, 'Zn': Zn}
out_of_range = [f"**{k}** = {v} (trained range: {ranges[k][0]}–{ranges[k][1]} kg)"
                for k, v in inputs.items() if not (ranges[k][0] <= v <= ranges[k][1])]

if out_of_range:
    st.warning("⚠️ Some inputs are outside the trained range — results may be less reliable:\n\n" +
               "\n".join(f"- {w}" for w in out_of_range))

# ── Predict Button ─────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🔍 Predict Environmental Impact", use_container_width=True):

    env_inputs = pd.DataFrame([[N, P, K, Zn]], columns=['N_rate', 'P_rate', 'K_rate', 'Zn_rate'])
    eco_inputs = pd.DataFrame([[Zn]], columns=['Zn_rate'])

    env_pred = model1.predict(env_inputs)[0]
    eco_pred = model2.predict(eco_inputs)[0]

    # ── Results ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Predicted Environmental Impact Scores")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🌍 Global Warming",            f"{env_pred[0]:,.2f} kg CO₂-eq")
        st.metric("💧 Freshwater Eutrophication", f"{env_pred[1]:.6f} kg P-eq")

    with col2:
        st.metric("🌫️ Terrestrial Acidification", f"{env_pred[2]:.4f} kg SO₂-eq")
        st.metric("☠️ Terrestrial Ecotoxicity",   f"{eco_pred:,.2f} CTUe")

    st.markdown("---")
    st.caption("Model trained on conventional rice cultivation LCA data using ecoinvent processes. "
               "Results are predictions — not a substitute for full LCA.")