import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ---------------- SETTINGS ---------------- #
data_path = r"C:\Users\scvst\Desktop\ML Project\Data\Trial Run 2 ML.csv"
save_dir  = r"C:\Users\scvst\Desktop\ML Project"

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv(data_path)

# ---------------- FEATURES (NPK ONLY) ---------------- #
X = df[['N_rate', 'P_rate', 'K_rate']]
y = df[['global_warming',
        'freshwater_eutrophication',
        'terrestrial_acidification']]

targets = [
    'Global Warming',
    'Freshwater Eutrophication',
    'Terrestrial Acidification'
]

units = ['kg CO₂-eq', 'kg P-eq', 'kg SO₂-eq']

# ---------------- TRAIN ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------- STYLE ---------------- #
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11
})

# =========================================================
# 🔥 1. INDIVIDUAL MODEL VALIDATION PLOTS
# =========================================================
for i in range(3):
    actual = y_test.iloc[:, i]
    pred   = y_pred[:, i]

    r2 = r2_score(actual, pred)

    mn = min(actual.min(), pred.min())
    mx = max(actual.max(), pred.max())

    plt.figure(figsize=(6,5))

    plt.scatter(actual, pred,
                alpha=0.7,
                color='#1f77b4',
                edgecolors='white',
                s=60)

    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)

    plt.xlim(mn, mx)
    plt.ylim(mn, mx)

    plt.title(targets[i])
    plt.xlabel(f'Actual ({units[i]})')
    plt.ylabel(f'Predicted ({units[i]})')

    plt.text(0.05, 0.9, f'R² = {r2:.3f}', transform=plt.gca().transAxes)

    filename = f"validation_{targets[i].replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

# =========================================================
# 🔥 2. COMBINED VALIDATION PLOT
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, ax in enumerate(axes):
    actual = y_test.iloc[:, i]
    pred   = y_pred[:, i]

    r2 = r2_score(actual, pred)

    mn = min(actual.min(), pred.min())
    mx = max(actual.max(), pred.max())

    ax.scatter(actual, pred,
               alpha=0.7,
               color='#1f77b4',
               edgecolors='white',
               s=40)

    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)

    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)

    ax.set_title(targets[i])
    ax.text(0.05, 0.9, f'R² = {r2:.3f}', transform=ax.transAxes)

plt.suptitle("Model Validation (NPK → Impacts)", fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "validation_combined.png"), dpi=300)
plt.close()

# =========================================================
# 🔥 3. FEATURE IMPORTANCE (NPK ONLY)
# =========================================================
features = ['N', 'P', 'K']
importances = []

for i in range(3):
    m = RandomForestRegressor(n_estimators=100, random_state=42)
    m.fit(X, y.iloc[:, i])
    importances.append(m.feature_importances_)

importances = np.array(importances)

x = np.arange(len(features))
width = 0.25

plt.figure(figsize=(7,5))

labels = ['GW', 'FE', 'TA']
colors = ['#2ecc71', '#3498db', '#e67e22']

for i in range(3):
    plt.bar(x + i*width, importances[i], width,
            label=labels[i], color=colors[i])

plt.xticks(x + width, features)
plt.ylabel('Importance')
plt.title('Feature Importance (NPK)')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "feature_importance_npk.png"), dpi=300)
plt.close()

# =========================================================
# 🔥 4. NORMALIZED OVERALL METRICS
# =========================================================
scaler = StandardScaler()

y_true_scaled = scaler.fit_transform(y_test)
y_pred_scaled = scaler.transform(y_pred)

y_true_all = y_true_scaled.flatten()
y_pred_all = y_pred_scaled.flatten()

r2 = r2_score(y_true_all, y_pred_all)
rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
mae = mean_absolute_error(y_true_all, y_pred_all)

print("\n📊 Overall Model Performance (Normalized)")
print(f"R²   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")

print("\n✅ Done! All plots saved in project folder.")