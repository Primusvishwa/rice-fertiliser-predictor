# 🌱 Rice Fertiliser Impact Predictor

A Machine Learning tool that predicts the environmental impact of fertiliser application in rice cultivation — without requiring a full Life Cycle Assessment (LCA) or access to expensive databases like ecoinvent.

🔗 **Live App:** [rice-lca-predictor.streamlit.app](https://rice-lca-predictor.streamlit.app)

---

## 🎯 Motivation

Life Cycle Assessment is the gold standard for measuring environmental impact of agricultural practices. However, it requires:
- Expensive software and databases (e.g. ecoinvent)
- Technical expertise
- Hours of manual setup

This project makes LCA-based insights accessible to **anyone** — farmers, researchers, students — by training ML models on LCA outputs and wrapping them in a simple web interface.

---

## 🌾 Scope

- **Crop:** Conventional rice cultivation
- **Inputs:** Nitrogen (N), Phosphorus (P), Potassium (K), Zinc (Zn) fertiliser rates
- **Impact Categories:**
  - Global Warming Potential (GWP) — kg CO₂-eq
  - Freshwater Eutrophication — kg P-eq
  - Terrestrial Acidification — kg SO₂-eq
  - Terrestrial Ecotoxicity — CTUe

---

## 🤖 Machine Learning Approach

### Data Generation
- LCA simulations run in **OpenLCA** using ecoinvent processes
- Parametric sampling across fertiliser input ranges:
  - N: 120–150 kg/ha
  - P: 40–60 kg/ha
  - K: 30–40 kg/ha
  - Zn: 10–30 kg/ha
- 1000 samples using OAT (One-At-a-Time) and RANDOM sampling

### Models
| Model | Inputs | Targets | Algorithm | R² |
|---|---|---|---|---|
| Model 1 | N, P, K, Zn | GWP, Eutrophication, Acidification | Random Forest | 0.96+ |
| Model 2 | Zn | Ecotoxicity | Linear Regression | 0.9998 |

### Key Finding
Feature importance analysis revealed that **Zn exclusively drives Ecotoxicity** (importance = 1.0), while N dominates the remaining three impact categories (~75% importance). This led to a two-model architecture for optimal performance.

---

## 📁 Repository Structure
```
rice-fertiliser-predictor/
│
├── app.py                  ← Streamlit web application
├── requirements.txt        ← Python dependencies
├── model1_env_impacts.pkl  ← Trained Random Forest model
├── model2_ecotoxicity.pkl  ← Trained Linear Regression model
│
├── Scripts/
│   ├── evaluate.py         ← Train and evaluate both models
│   ├── predict.py          ← Terminal-based predictor
│   ├── visualise.py        ← Generate validation plots
│   ├── cross_validate.py   ← Cross validation
│   └── retrain.py          ← Retrain on new data
│
├── Data/
│   └── Trial Run 2 ML.csv  ← Training dataset
│
└── Plots/
    ├── plot1_predicted_vs_actual.png
    ├── plot2_feature_importance.png
    └── plot3_input_vs_impact.png
```

---

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠 Tech Stack

- **Python** — core language
- **OpenLCA** — LCA simulation and data generation
- **scikit-learn** — ML models
- **pandas / numpy** — data processing
- **matplotlib** — visualisations
- **Streamlit** — web application

---

## 👤 Author

**Primusvishwa**
B.Tech Biotechnology
*College Project — Environmental Impact Prediction using Machine Learning*
```
