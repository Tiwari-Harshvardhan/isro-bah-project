import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============ 1. LOAD DATA ============
df = pd.read_csv('../data/final/final_mp_dataset.csv') 

print("Dataset loaded successfully!")
print(f"Data shape: {df.shape}")
print(f"Districts: {df['district'].nunique()}")
print("\nMissing values per column:")
print(df.isnull().sum())
df = df.dropna()
print(f"\nData shape after removing NaN: {df.shape}")

# ============ 2. DEFINE MAPE FUNCTION ============
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE (Mean Absolute Percentage Error)"""
    mask = y_true != 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    if len(y_true_filtered) == 0:
        return np.nan
    
    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100

# ============ 3. PREPARE FEATURES AND TARGET ============
feature_columns = [
    'year',
    'population_density',
    'ndvi',
    'ndvi_builtup_interaction',
    'built_up_growth',
    'ndvi_anomally',
    'lst_anomally',
    'night_lst',
    'total_population',
    'built_up_pct',
    'month_sin',
    'month_cos',
    'lst_lag1',
    'lst_lag2',
    'lst_lag3',
    'lst_3month_avg',
    'lst_12month_avg'
]

X = df[feature_columns]
y = df['target_lst'] 

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print("\nFeature names:")
print(X.columns.tolist())

# ============ 4. CHRONOLOGICAL SPLIT ============
train_mask = df['year'].between(2018, 2023)
val_mask = df['year'] == 2024
test_mask = df['year'] == 2025

X_train = X[train_mask]
y_train = y[train_mask]
X_val = X[val_mask]
y_val = y[val_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"\nData Split Summary:")
print(f"Training set: {X_train.shape[0]} samples (2018-2023)")
print(f"Validation set: {X_val.shape[0]} samples (2024)")
print(f"Test set: {X_test.shape[0]} samples (2025)")

# ============ 5. FEATURE SCALING ============
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)

# ============ 6. TRAIN ENHANCED MODEL ============
print("\n" + "="*50)
print("Training Enhanced Linear Regression Model")
print("="*50)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Model training completed!")
print(f"Intercept: {model.intercept_:.4f}")

# ============ 7. FEATURE COEFFICIENTS ============
coef_df = pd.DataFrame({
    'feature': feature_columns,
    'coefficient': model.coef_
})
coef_df['abs_coefficient'] = np.abs(coef_df['coefficient'])
coef_df = coef_df.sort_values('abs_coefficient', ascending=False)

print("\nFeature Coefficients:")
print(coef_df)

# Plot feature coefficients
plt.figure(figsize=(12, 8))
colors = ['red' if c < 0 else 'green' for c in coef_df['coefficient']]
plt.barh(coef_df['feature'], coef_df['coefficient'], color=colors)
plt.xlabel('Coefficient Value')
plt.title('Linear Regression Feature Coefficients (Enhanced Model)')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_regression_coefficients.png', dpi=300)
plt.show()

# ============ 8. PREDICTIONS WITH ENHANCED MODEL ============
print("\nMaking predictions with enhanced model...")
y_pred_train = model.predict(X_train_scaled)
y_pred_val = model.predict(X_val_scaled)
y_pred_test = model.predict(X_test_scaled)

# ============ 9. EVALUATION FUNCTION ============
def evaluate_predictions(y_true, y_pred, dataset_name):
    """Calculate and print evaluation metrics including MAPE"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    print(f"\n{dataset_name} Metrics:")
    print(f"RMSE: {rmse:.4f}°C")
    print(f"MAE:  {mae:.4f}°C")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²:   {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}

# Evaluate enhanced model
metrics_train = evaluate_predictions(y_train, y_pred_train, "Training (Enhanced)")
metrics_val = evaluate_predictions(y_val, y_pred_val, "Validation (Enhanced)")
metrics_test = evaluate_predictions(y_test, y_pred_test, "Test (Enhanced)")

# ============ 10. TRAIN BASELINE MODEL FOR COMPARISON ============
print("\n" + "="*50)
print("Training Baseline Model (Original Features Only)")
print("="*50)

# Define original features (without engineered features)
original_features = [
    'ndvi',
    'night_lst',
    'total_population',
    'built_up_pct',
    'month_sin',
    'month_cos',
    'lst_lag1',
    'lst_lag2',
    'lst_lag3',
    'lst_3month_avg',
    'lst_12month_avg'
]

X_original = df[original_features]
X_original_train = X_original[train_mask]
X_original_test = X_original[test_mask]

# Scale original features
scaler_orig = StandardScaler()
X_orig_train_scaled = scaler_orig.fit_transform(X_original_train)
X_orig_test_scaled = scaler_orig.transform(X_original_test)

# Train baseline model
model_orig = LinearRegression()
model_orig.fit(X_orig_train_scaled, y_train[train_mask])
y_pred_orig_test = model_orig.predict(X_orig_test_scaled)

# Evaluate baseline model
rmse_orig = np.sqrt(mean_squared_error(y_test, y_pred_orig_test))
mae_orig = mean_absolute_error(y_test, y_pred_orig_test)
r2_orig = r2_score(y_test, y_pred_orig_test)
mape_orig = mean_absolute_percentage_error(y_test, y_pred_orig_test)

print(f"\nBaseline Model Test Metrics:")
print(f"RMSE: {rmse_orig:.4f}°C")
print(f"MAE:  {mae_orig:.4f}°C")
print(f"MAPE: {mape_orig:.2f}%")
print(f"R²:   {r2_orig:.4f}")

# ============ 11. CALCULATE IMPROVEMENTS ============
rmse_enhanced = metrics_test['rmse']
mae_enhanced = metrics_test['mae']
r2_enhanced = metrics_test['r2']
mape_enhanced = metrics_test['mape']

rmse_improvement = ((rmse_orig - rmse_enhanced) / rmse_orig) * 100
mae_improvement = ((mae_orig - mae_enhanced) / mae_orig) * 100
mape_improvement = ((mape_orig - mape_enhanced) / mape_orig) * 100
r2_improvement = (r2_enhanced - r2_orig) * 100

print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
print(f"RMSE Improvement: {rmse_improvement:.2f}%")
print(f"MAE Improvement:  {mae_improvement:.2f}%")
print(f"MAPE Improvement: {mape_improvement:.2f}%")
print(f"R² Improvement:   {r2_improvement:.2f}%")

# ============ 12. VISUALIZATIONS ============

# Plot 1: Actual vs Predicted (All sets)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

datasets = [
    (y_train, y_pred_train, "Training", metrics_train),
    (y_val, y_pred_val, "Validation", metrics_val),
    (y_test, y_pred_test, "Test", metrics_test)
]

for ax, (y_true, y_pred, name, metrics) in zip(axes, datasets):
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Actual LST (°C)')
    ax.set_ylabel('Predicted LST (°C)')
    ax.set_title(f'{name} Set\nR² = {metrics["r2"]:.4f}, RMSE = {metrics["rmse"]:.2f}°C')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_predictions.png', dpi=300)
plt.show()

# Plot 2: Residuals Analysis
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (y_true, y_pred, name, _) in zip(axes, datasets):
    residuals = y_true - y_pred
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Residuals (°C)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{name} Set Residuals\nMean = {residuals.mean():.4f}°C')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_residuals.png', dpi=300)
plt.show()

# Plot 3: Time Series Predictions (Sample District)
sample_district = df['district'].iloc[0]
district_mask = df['district'] == sample_district
district_data = df[district_mask].sort_values(['year', 'month'])
district_indices = district_mask.values

district_pred = model.predict(scaler.transform(X[district_indices]))

plt.figure(figsize=(14, 6))
time_idx = range(len(district_data))
plt.plot(time_idx, y[district_indices], label='Actual LST', alpha=0.8, linewidth=2)
plt.plot(time_idx, district_pred, label='Predicted LST', alpha=0.8, linestyle='--', linewidth=2)
plt.xlabel('Time (Months)')
plt.ylabel('LST (°C)')
plt.title(f'Time Series: Actual vs Predicted LST - {sample_district}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_regression_timeseries.png', dpi=300)
plt.show()

# Plot 4: Residuals vs Predicted (Test Set)
plt.figure(figsize=(10, 6))
residuals_test = y_test - y_pred_test
plt.scatter(y_pred_test, residuals_test, alpha=0.5, s=20)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted LST (°C)')
plt.ylabel('Residuals (°C)')
plt.title('Test Set: Residuals vs Predicted Values')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_regression_residuals_vs_predicted.png', dpi=300)
plt.show()

# Plot 5: Correlation Matrix
plt.figure(figsize=(14, 12))
correlation_matrix = df[feature_columns + ['target_lst']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
plt.show()

# Plot 6: Model Performance Comparison (Bar Chart)
plt.figure(figsize=(12, 6))
metrics_to_plot = ['RMSE', 'MAE', 'MAPE']
values_orig = [rmse_orig, mae_orig, mape_orig]
values_enhanced = [rmse_enhanced, mae_enhanced, mape_enhanced]

x = np.arange(len(metrics_to_plot))
width = 0.35

bars1 = plt.bar(x - width/2, values_orig, width, label='Baseline Model', alpha=0.7, color='skyblue')
bars2 = plt.bar(x + width/2, values_enhanced, width, label='Enhanced Model', alpha=0.7, color='lightcoral')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Model Performance Comparison: Baseline vs Enhanced')
plt.xticks(x, metrics_to_plot)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300)
plt.show()

# Plot 7: Performance Improvement Visualization
plt.figure(figsize=(10, 6))
improvements = [rmse_improvement, mae_improvement, mape_improvement, r2_improvement]
improvement_labels = ['RMSE', 'MAE', 'MAPE', 'R²']
colors_imp = ['green' if x > 0 else 'red' for x in improvements]

bars = plt.bar(improvement_labels, improvements, color=colors_imp, alpha=0.7)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

plt.xlabel('Metrics')
plt.ylabel('Improvement (%)')
plt.title('Performance Improvement with Feature Engineering')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('performance_improvement.png', dpi=300)
plt.show()

# Plot 8: Q-Q plot for residuals (Test set)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
stats.probplot(residuals_test, dist="norm", plot=plt)
plt.title('Q-Q Plot of Test Set Residuals')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_pred_test, residuals_test, alpha=0.5, s=20)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_diagnostics.png', dpi=300)
plt.show()

# ============ 13. SAVE RESULTS ============
# Save predictions
results_df = pd.DataFrame({
    'district': df['district'],
    'year': df['year'],
    'month': df['month'],
    'actual_lst': y,
    'predicted_lst': model.predict(scaler.transform(X)),
    'residual': y - model.predict(scaler.transform(X))
})
results_df.to_csv('linear_regression_predictions.csv', index=False)
print("\nPredictions saved to 'linear_regression_predictions.csv'")

# Save model coefficients
model_params = pd.DataFrame({
    'feature': feature_columns,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
})
model_params.to_csv('linear_regression_coefficients.csv', index=False)
print("Coefficients saved to 'linear_regression_coefficients.csv'")

# Save scaler parameters
scaler_params = pd.DataFrame({
    'feature': feature_columns,
    'mean': scaler.mean_,
    'scale': scaler.scale_
})
scaler_params.to_csv('scaler_parameters.csv', index=False)
print("Scaler parameters saved to 'scaler_parameters.csv'")

# Save metrics comparison
metrics_comparison = pd.DataFrame({
    'Model': ['Baseline', 'Enhanced'],
    'RMSE': [rmse_orig, rmse_enhanced],
    'MAE': [mae_orig, mae_enhanced],
    'MAPE': [mape_orig, mape_enhanced],
    'R2': [r2_orig, r2_enhanced]
})
metrics_comparison.to_csv('model_metrics_comparison.csv', index=False)
print("Metrics comparison saved to 'model_metrics_comparison.csv'")

# ============ 14. FINAL SUMMARY REPORT ============
print("\n" + "="*50)
print("FINAL MODEL SUMMARY REPORT")
print("="*50)
print(f"Model Type: Linear Regression with Engineered Features")
print(f"Number of Features: {len(feature_columns)}")
print(f"Training Samples: {len(X_train)}")
print(f"Validation Samples: {len(X_val)}")
print(f"Test Samples: {len(X_test)}")
print(f"\nIntercept: {model.intercept_:.4f}")
print("\nTop 5 Most Important Features:")
for idx, row in coef_df.head(5).iterrows():
    print(f"  {row['feature']:25s}: {row['coefficient']:+.4f} (abs: {row['abs_coefficient']:.4f})")

print("\n" + "-"*50)
print("TEST SET PERFORMANCE:")
print("-"*50)
print(f"RMSE: {metrics_test['rmse']:.4f}°C")
print(f"MAE:  {metrics_test['mae']:.4f}°C")
print(f"MAPE: {metrics_test['mape']:.2f}%")
print(f"R²:   {metrics_test['r2']:.4f}")

print("\n" + "-"*50)
print("IMPROVEMENT OVER BASELINE MODEL:")
print("-"*50)
print(f"RMSE: {rmse_improvement:+.2f}% (Baseline: {rmse_orig:.4f}°C → Enhanced: {rmse_enhanced:.4f}°C)")
print(f"MAE:  {mae_improvement:+.2f}% (Baseline: {mae_orig:.4f}°C → Enhanced: {mae_enhanced:.4f}°C)")
print(f"MAPE: {mape_improvement:+.2f}% (Baseline: {mape_orig:.2f}% → Enhanced: {mape_enhanced:.2f}%)")
print(f"R²:   {r2_improvement:+.2f}% (Baseline: {r2_orig:.4f} → Enhanced: {r2_enhanced:.4f})")
print("="*50)

print("\nModel training and evaluation complete!")