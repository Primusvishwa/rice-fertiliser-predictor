# Train only on RANDOM rows and test only on RANDOM rows
df_random = df[df['sample_type'] == 'RANDOM']

X_r = df_random[['N_rate', 'P_rate', 'K_rate', 'Zn_rate']]
y_r = df_random[['global_warming', 'freshwater_eutrophication',
                  'terrestrial_acidification', 'terrestrial_ecotoxicity']]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_r, y_r, test_size=0.2, random_state=42
)

model_r = RandomForestRegressor(n_estimators=100, random_state=42)
model_r.fit(X_train_r, y_train_r)
y_pred_r = model_r.predict(X_test_r)

print("=== RANDOM-only Results ===")
for i, name in enumerate(['Global Warming', 'Freshwater Eutrophication',
                           'Terrestrial Acidification', 'Terrestrial Ecotoxicity']):
    r2   = r2_score(y_test_r.iloc[:, i], y_pred_r[:, i])
    mae  = mean_absolute_error(y_test_r.iloc[:, i], y_pred_r[:, i])
    rmse = np.sqrt(mean_squared_error(y_test_r.iloc[:, i], y_pred_r[:, i]))
    print(f"{name:<30} R²: {r2:.4f}  MAE: {mae:.4f}  RMSE: {rmse:.4f}")