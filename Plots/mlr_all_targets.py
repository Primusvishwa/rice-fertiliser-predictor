import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load & Clean Data ──────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\scvst\Desktop\ML Project\Data\Trial Run 2 ML.csv")
df = df.drop(columns=["sample_type"], errors="ignore").dropna()
print(f"Dataset shape: {df.shape}")
print(df.describe().round(3))

# ── 2. Define Inputs & Targets ────────────────────────────────────────────────
FEATURES = ["N_rate", "P_rate", "K_rate", "Zn_rate"]

TARGETS = {
    "global_warming":              "Global Warming Potential (GWP)",
    "freshwater_eutrophication":   "Freshwater Eutrophication",
    "terrestrial_acidification":   "Terrestrial Acidification",
    "terrestrial_ecotoxicity":     "Terrestrial Ecotoxicity",
}

COLORS = {
    "global_warming":            ("#2E86AB", "viridis",  "#E07A5F", "#2A9D8F"),
    "freshwater_eutrophication": ("#457B9D", "plasma",   "#E76F51", "#2A9D8F"),
    "terrestrial_acidification": ("#6A4C93", "cividis",  "#F4A261", "#2A9D8F"),
    "terrestrial_ecotoxicity":   ("#1D3557", "inferno",  "#E07A5F", "#2A9D8F"),
}

# ── 3. Loop Over Each Target ──────────────────────────────────────────────────
all_results = {}

for target, label in TARGETS.items():

    print("\n" + "="*65)
    print(f"  TARGET: {label}")
    print("="*65)

    X = df[FEATURES]
    y = df[target]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Fit
    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    # Predict
    y_pred_train = model.predict(X_train_sc)
    y_pred_test  = model.predict(X_test_sc)

    # Metrics
    rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_tr  = mean_absolute_error(y_train, y_pred_train)
    r2_tr   = r2_score(y_train, y_pred_train)

    rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_te  = mean_absolute_error(y_test, y_pred_test)
    r2_te   = r2_score(y_test, y_pred_test)

    print(f"[Train]  R²={r2_tr:.4f}  RMSE={rmse_tr:.4f}  MAE={mae_tr:.4f}")
    print(f"[Test ]  R²={r2_te:.4f}  RMSE={rmse_te:.4f}  MAE={mae_te:.4f}")

    # Regression equation (original units)
    coef_orig      = model.coef_ / scaler.scale_
    intercept_orig = model.intercept_ - np.dot(coef_orig, scaler.mean_)
    eq = f"{label} = {intercept_orig:.4f}"
    for feat, c in zip(FEATURES, coef_orig):
        sign = "+" if c >= 0 else "-"
        eq += f" {sign} {abs(c):.4f}·{feat}"
    print(f"\nEquation: {eq}")

    # Store
    all_results[target] = dict(
        model=model, scaler=scaler,
        X_test=X_test, y_test=y_test,
        y_pred_test=y_pred_test,
        coef_orig=coef_orig, intercept_orig=intercept_orig,
        r2_tr=r2_tr, rmse_tr=rmse_tr, mae_tr=mae_tr,
        r2_te=r2_te, rmse_te=rmse_te, mae_te=mae_te,
        eq=eq, label=label
    )

# ── 4. Generate One Figure Per Target ────────────────────────────────────────
for target, label in TARGETS.items():
    r  = all_results[target]
    c_scatter, c_cmap, c_trend, c_hist = COLORS[target]

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"Multivariate Linear Regression\nN, P, K, Zn  →  {label}",
                 fontsize=15, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.60, wspace=0.42)

    # ── Panel A: Actual vs Predicted ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :3])
    sc = ax1.scatter(r["y_test"], r["y_pred_test"], alpha=0.55,
                     c=r["X_test"]["Zn_rate"], cmap=c_cmap,
                     edgecolors="white", linewidths=0.4, s=50)
    plt.colorbar(sc, ax=ax1, label="Zn_rate")
    lims = [min(r["y_test"].min(), r["y_pred_test"].min()),
            max(r["y_test"].max(), r["y_pred_test"].max())]
    ax1.plot(lims, lims, "r--", lw=1.8, label="Perfect fit")
    ax1.set_xlabel(f"Actual {label}", fontsize=11)
    ax1.set_ylabel(f"Predicted {label}", fontsize=11)
    ax1.set_title(f"Actual vs Predicted  (Test R² = {r['r2_te']:.4f})", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # ── Panel B: Feature Importance ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 3])
    coefs = r["model"].coef_
    bar_colors = ["#E07A5F" if v < 0 else "#3D405B" for v in coefs]
    bars = ax2.barh(FEATURES, coefs, color=bar_colors, edgecolor="white", height=0.45)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Std. Coefficient", fontsize=10)
    ax2.set_title("Feature Importance\n(Standardised)", fontsize=11)
    for bar, val in zip(bars, coefs):
        ax2.text(val + (0.3 if val >= 0 else -0.3),
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center",
                 ha="left" if val >= 0 else "right", fontsize=9)
    ax2.grid(axis="x", alpha=0.3)

    # ── Panel C: Stats Table ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.axis("off")
    table_data = [
        ["Metric",  "Train",                    "Test"],
        ["R²",      f"{r['r2_tr']:.4f}",        f"{r['r2_te']:.4f}"],
        ["RMSE",    f"{r['rmse_tr']:.4f}",       f"{r['rmse_te']:.4f}"],
        ["MAE",     f"{r['mae_tr']:.4f}",        f"{r['mae_te']:.4f}"],
        ["N samples", f"{int(len(r['y_test'])*4)}", f"{len(r['y_test'])}"],
    ]
    tbl = ax3.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.4, 2.0)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#3D405B")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F0F0F0")
        cell.set_edgecolor("#CCCCCC")
    ax3.set_title("Model Performance Metrics", fontsize=12, fontweight="bold", pad=10)

    # ── Panel D: Regression Equation ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.axis("off")
    ic = r["intercept_orig"]
    eq_lines = [f"{label} =", f"  {ic:.4f}"]
    for feat, c in zip(FEATURES, r["coef_orig"]):
        sign = "+" if c >= 0 else "-"
        eq_lines.append(f"  {sign} {abs(c):.4f} × {feat}")
    eq_text = "\n".join(eq_lines)
    ax4.text(0.05, 0.55, eq_text, transform=ax4.transAxes,
             fontsize=10.5, verticalalignment="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#EAF4FB",
                       edgecolor="#2E86AB", linewidth=1.5))
    ax4.set_title("Regression Equation (Original Units)", fontsize=12,
                  fontweight="bold")

    # ── Panels E–H: Partial Effect Plots ─────────────────────────────────────
    means = r["X_test"].mean()
    for idx, feat in enumerate(FEATURES):
        ax = fig.add_subplot(gs[2, idx])
        ax.scatter(r["X_test"][feat], r["y_test"], alpha=0.4,
                   color="#81B29A", edgecolors="white",
                   linewidths=0.3, s=25, label="Actual")
        x_line = np.linspace(r["X_test"][feat].min(),
                             r["X_test"][feat].max(), 200)
        X_line = pd.DataFrame({f: [means[f]] * 200 for f in FEATURES})
        X_line[feat] = x_line
        y_line = r["model"].predict(r["scaler"].transform(X_line))
        ax.plot(x_line, y_line, color=c_trend, lw=2, label="Model trend")
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel(label[:18] if idx == 0 else "", fontsize=9)
        ax.set_title(f"{feat} Effect\n(others @ mean)", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # ── Save ──────────────────────────────────────────────────────────────────
    safe_name = target.replace(" ", "_")
    out = rf"C:\Users\scvst\Desktop\ML Project\Data\MLR_{safe_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")

print("\n✅ All 4 figures generated and saved!")
