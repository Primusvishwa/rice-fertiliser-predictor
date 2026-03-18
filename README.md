# 🌱 Rice Fertiliser Impact Predictor

A Machine Learning tool that predicts the environmental impact of fertiliser application in rice cultivation — without requiring a full Life Cycle Assessment (LCA) or access to expensive databases like ecoinvent.

🔗 **Live App:** [rice-lca-predictor.streamlit.app](https://rice-lca-predictor.streamlit.app)

---

## 🎯 Motivation

Life Cycle Assessment is the gold standard for measuring environmental impact of agricultural practices. However, it requires:
- Expensive software and databases (e.g. ecoinvent)
- Technical expertise
- Hours of manual setup

This project makes LCA-based insights accessible to **anyone** — farmers, researchers, students — by training an ML model on LCA outputs and wrapping it in a simple web interface.

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

### Model

| Inputs | Targets | Algorithm | R² |
|---|---|---|---|
| N, P, K, Zn | GWP, Eutrophication, Acidification, Ecotoxicity | Multivariate Linear Regression | 1.0000 |

### Key Finding
Feature importance analysis revealed that the relationships between fertiliser inputs and all four LCA impact categories are **perfectly linear** within the trained input ranges. A single **Multivariate Linear Regression** model with N, P, K, Zn as inputs accurately predicts all four impact categories simultaneously with R² = 1.0000.

---

## 🌾 Field Emission Calculator

In addition to the LCA Impact Predictor, the tool includes a **Field Emission Calculator** based on established scientific methodologies:

| Emission | Formula | Methodology |
|---|---|---|
| Methane (CH₄) | EFc × Duration × SFo × SFw | IPCC |
| Nitrous Oxide (N₂O) | (FSN + FON) × EF × 44/28 | IPCC |
| Nitrate (NO₃) | N_total × LF × 62/14 | IPCC |
| Ammonia (NH₃) | N_total × EF × 17/14 | IPCC |
| Phosphate (PO₄) | P_total × RF × 95/31 | SALCA |

Supports two irrigation modes (Fully Flooded / AWD) and two organic amendments (Farm-Yard Manure / Vermicompost) with automatic N and P contribution calculations.

---

## 📁 Repository Structure
```
rice-fertiliser-predictor/
│
├── app.py                      ← Streamlit web application
├── requirements.txt            ← Python dependencies
├── model_all_impacts.pkl       ← Trained MLR model (all 4 impact categories)
│
├── Scripts/
│   ├── evaluate.py             ← Train and evaluate model
│   ├── predict.py              ← Terminal-based predictor
│   ├── visualise.py            ← Generate validation plots
│   ├── cross_validate.py       ← Cross validation
│   └── retrain.py              ← Retrain on new data
│
├── Data/
│   └── Trial Run 2 ML.csv     ← Training dataset
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
