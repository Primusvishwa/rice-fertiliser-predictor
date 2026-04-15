import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\scvst\Desktop\ML Project\Data\Trial Run 2 ML.csv")

# ── MODEL: N, P, K → GWP + Eutrophication + Acidification ────────────────────
X1 = df[['N_rate', 'P_rate', 'K_rate']]
y1 = df[['global_warming', 'freshwater_eutrophication', 'terrestrial_acidification']]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

model1 = RandomForestRegressor(n_estimators=100, random_state=42)
model1.fit(X1_train, y1_train)
y1_pred = model1.predict(X1_test)

print("=" * 65)
print("MODEL — N, P, K  →  Environmental Impacts")
print(f"{'Impact Category':<30} {'R²':>8} {'MAE':>12} {'RMSE':>12}")
print("=" * 65)
for i, name in enumerate(['Global Warming', 'Freshwater Eutrophication', 'Terrestrial Acidification']):
    r2   = r2_score(y1_test.iloc[:, i], y1_pred[:, i])
    mae  = mean_absolute_error(y1_test.iloc[:, i], y1_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y1_test.iloc[:, i], y1_pred[:, i]))
    print(f"{name:<30} {r2:>8.4f} {mae:>12.4f} {rmse:>12.4f}")
print("=" * 65)

# ── Save Model ─────────────────────────────────────────────────────────────────
joblib.dump(model1, r"C:\Users\scvst\Desktop\ML Project\model_all_impacts.pkl")
print("\n✅ Model saved as model_all_impacts.pkl — ready to deploy!")