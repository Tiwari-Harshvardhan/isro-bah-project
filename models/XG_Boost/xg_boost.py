import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path
import joblib
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============ 1. LOAD DATA ============
df = pd.read_csv('C://Users//harsh//Desktop//isro hacakthon//data//final//Delhi_wardwise_lst_and_ndvi_and_builtup_pop_2018_to_2025.csv')

print("Dataset loaded successfully!")
print(f"Data shape: {df.shape}")
print(f"Wards: {df['ward'].nunique()}")
print(f"Date range: {df['year'].min()} - {df['year'].max()}")

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Remove rows with NaN values(only where target is missing)
df = df[df['mean_lst_day_celsius'].notna()]
print(f"\nData shape after removing rows with missing target: {df.shape}")

# ============ HANDLE MISSING VALUES IN FEATURES ============
print("\n" + "="*50)
print("HANDLING MISSING VALUES IN FEATURES")
print("="*50)

# For each feature column, apply appropriate imputation
feature_columns_with_handling = {
    'ndvi_lag1': 'forward_fill',
    'ndvi_lag2': 'forward_fill',
    'ndvi_3month_avg': 'rolling_mean',
    'lst_3month_avg': 'rolling_mean',
    'lst_12month_avg': 'rolling_mean',
    'ndvi_change': 'forward_fill',
    'builtup_ndvi': 'interaction',
}

# Group by ward for time series imputation
for ward in df['ward'].unique():
    ward_mask = df['ward'] == ward
    ward_data = df[ward_mask].sort_values(['year', 'month'])
    
    # Forward fill lagged features
    for col in ['ndvi_lag1', 'ndvi_lag2', 'ndvi_change']:
        if col in df.columns:
            df.loc[ward_mask, col] = ward_data[col].fillna(method='ffill')
    
    # Fill remaining NaN with 0 (for features where 0 makes sense)
    for col in ['ndvi_lag1', 'ndvi_lag2', 'ndvi_change']:
        if col in df.columns:
            df.loc[ward_mask, col] = df.loc[ward_mask, col].fillna(0)

# For rolling averages, fill with ward-wise mean
for col in ['ndvi_3month_avg', 'lst_3month_avg', 'lst_12month_avg']:
    if col in df.columns:
        # First try forward fill within each ward
        for ward in df['ward'].unique():
            ward_mask = df['ward'] == ward
            df.loc[ward_mask, col] = df.loc[ward_mask, col].fillna(method='ffill')
        
        # Fill remaining NaN with ward-wise mean
        ward_means = df.groupby('ward')[col].mean()
        for ward in df['ward'].unique():
            ward_mask = df['ward'] == ward
            if not ward_means.loc[ward] is None:
                df.loc[ward_mask, col] = df.loc[ward_mask, col].fillna(ward_means.loc[ward])
        
        # Final fallback: fill with global mean
        df[col] = df[col].fillna(df[col].mean())

# For builtup_ndvi (interaction term), recompute if missing
if 'builtup_ndvi' in df.columns:
    # If builtup_ndvi is missing but built_up_percent and mean_ndvi exist
    mask = df['builtup_ndvi'].isna()
    if mask.any():
        df.loc[mask, 'builtup_ndvi'] = (
            df.loc[mask, 'built_up_percent'] * df.loc[mask, 'mean_ndvi']
        )

# For urban_intensity and heat_exposure_index, forward fill
for col in ['urban_intensity', 'heat_exposure_index']:
    if col in df.columns:
        for ward in df['ward'].unique():
            ward_mask = df['ward'] == ward
            df.loc[ward_mask, col] = df.loc[ward_mask, col].fillna(method='ffill')
        df[col] = df[col].fillna(df[col].mean())

feature_columns = [
    'year',
    'year_index',
    'month',
    'population',
    'population_density',
    'area_sq_km',
    'built_up_percent',
    'mean_ndvi',
    'ndvi_lag1',
    'ndvi_lag2',
    'ndvi_3month_avg',
    'ndvi_change',
    'lst_3month_avg',
    'lst_12month_avg',
    'builtup_ndvi',
    'urban_intensity',
    'heat_exposure_index',
]

# Final check: count remaining missing values
print("\nMissing values after feature engineering:")
print(df[feature_columns].isnull().sum())
print(f"\nFinal data shape: {df.shape}")

X = df[feature_columns]
y = df['mean_lst_day_celsius']  # Target variable

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print("\nFeature names:")
print(X.columns.tolist())

# ============ 3. CHRONOLOGICAL SPLIT ============
# Data available: 2018-2022
# Split: 2018-2020 for training, 2021 for validation, 2022 for testing
train_mask = df['year'].between(2018, 2020)
val_mask = df['year'] == 2021
test_mask = df['year'] == 2022

X_train = X[train_mask]
y_train = y[train_mask]
X_val = X[val_mask]
y_val = y[val_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"\nData Split Summary:")
print(f"Training set: {X_train.shape[0]} samples (2018-2020)")
print(f"Validation set: {X_val.shape[0]} samples (2021)")
print(f"Test set: {X_test.shape[0]} samples (2022)")

# ============ 4. NO FEATURE SCALING ============
print("\nUsing raw features (XGBoost handles scaling internally)")

# ============ 5. HYPERPARAMETER TUNING WITH TIME SERIES CV ============
print("\n" + "="*50)
print("Hyperparameter Tuning with TimeSeriesSplit")
print("="*50)

# Create time series cross-validation
tscv = TimeSeriesSplit(n_splits=2)  
print(f"TimeSeriesSplit folds: {tscv.get_n_splits()}")

# Define parameter distribution for RandomizedSearch
param_dist = {
    'n_estimators': [200, 300, 500, 700, 1000],
    'max_depth': [3, 5, 7, 9, 11],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'colsample_bylevel': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5, 7],
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
    'reg_lambda': [0.1, 0.5, 1.0, 1.5, 2.0],
    'gamma': [0, 0.1, 0.2, 0.5, 1.0],
    'max_delta_step': [0, 1, 2]
}

# Base model - REMOVE early_stopping_rounds from here
base_model = xgb.XGBRegressor(
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# Randomized search with TimeSeriesSplit
print("Performing Randomized Search with TimeSeriesSplit (this may take several minutes)...")
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=25,  # Increased for better search
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1,
    error_score='raise'
)

random_search.fit(X_train, y_train)

print(f"\nBest parameters found:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest CV RMSE: {-random_search.best_score_:.4f}°C")

# ============ 6. TRAIN FINAL MODEL WITH EARLY STOPPING ============
print("\n" + "="*50)
print("Training Final Model with Early Stopping")
print("="*50)

# Use best parameters
best_params = random_search.best_params_.copy()

# Remove n_estimators from best_params to avoid duplication
if 'n_estimators' in best_params:
    best_params.pop('n_estimators')

# Create model with early stopping parameters in constructor
final_model = xgb.XGBRegressor(
    **best_params,
    n_estimators=2000,  # Start with large number
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# Train with early stopping on validation set
print("Training with early stopping on validation set...")
final_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

print(f"Best iteration: {final_model.best_iteration}")
print(f"Best score: {final_model.best_score:.4f}")

# Use the model
model = final_model

# ============ 7. FEATURE IMPORTANCE ============
# Multiple importance types from XGBoost booster
booster = model.get_booster()

# Get different importance types
importance_gain = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Get gain, cover, weight from native booster
try:
    importance_gain_native = pd.DataFrame({
        'feature': list(booster.get_score(importance_type='gain').keys()),
        'gain': list(booster.get_score(importance_type='gain').values())
    }).sort_values('gain', ascending=False)
    
    importance_weight = pd.DataFrame({
        'feature': list(booster.get_score(importance_type='weight').keys()),
        'weight': list(booster.get_score(importance_type='weight').values())
    }).sort_values('weight', ascending=False)
    
    importance_cover = pd.DataFrame({
        'feature': list(booster.get_score(importance_type='cover').keys()),
        'cover': list(booster.get_score(importance_type='cover').values())
    }).sort_values('cover', ascending=False)
    
    print("\nNative XGBoost Importance Types:")
    print("Gain (improvement in accuracy):")
    print(importance_gain_native.head(10))
    print("\nWeight (number of times feature used):")
    print(importance_weight.head(10))
    print("\nCover (average coverage of feature):")
    print(importance_cover.head(10))
    
except Exception as e:
    print(f"Native importance extraction failed: {e}")

print("\nFeature Importance (Built-in - Gain):")
print(importance_gain)

# Permutation Importance with more repeats
print("\nCalculating permutation importance (n_repeats=30)...")
perm_importance = permutation_importance(
    model, X_test, y_test, 
    n_repeats=30,
    random_state=42, 
    n_jobs=-1
)

perm_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}).sort_values('importance', ascending=False)

print("\nPermutation Importance:")
print(perm_importance_df)

# Plot feature importance
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Built-in importance
axes[0, 0].barh(importance_gain['feature'], importance_gain['importance'], color='blue', alpha=0.7)
axes[0, 0].set_xlabel('Importance (Gain)')
axes[0, 0].set_title('XGBoost Built-in Feature Importance')
axes[0, 0].grid(True, alpha=0.3)

# Permutation importance
axes[0, 1].barh(perm_importance_df['feature'], perm_importance_df['importance'], 
                color='green', alpha=0.7)
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_title('Permutation Importance (n_repeats=30)')
axes[0, 1].grid(True, alpha=0.3)

# Gain importance (native)
if 'importance_gain_native' in locals() and not importance_gain_native.empty:
    axes[1, 0].barh(importance_gain_native['feature'].head(15), 
                    importance_gain_native['gain'].head(15), 
                    color='orange', alpha=0.7)
    axes[1, 0].set_xlabel('Gain')
    axes[1, 0].set_title('Native XGBoost Gain Importance (Top 15)')
    axes[1, 0].grid(True, alpha=0.3)

# Weight importance (native)
if 'importance_weight' in locals() and not importance_weight.empty:
    axes[1, 1].barh(importance_weight['feature'].head(15), 
                    importance_weight['weight'].head(15), 
                    color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Weight (Frequency)')
    axes[1, 1].set_title('Native XGBoost Weight Importance (Top 15)')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xgboost_feature_importance.png', dpi=300)
plt.show()

# ============ 8. MAKE PREDICTIONS ============
print("\nMaking predictions...")
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

# Predict once and reuse
all_predictions = model.predict(X)

# ============ 9. EVALUATION FUNCTION ============
def evaluate_predictions(y_true, y_pred, dataset_name):
    """Calculate and print evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    medae = np.median(np.abs(y_true - y_pred))
    
    print(f"\n{dataset_name} Metrics:")
    print(f"RMSE: {rmse:.4f}°C")
    print(f"MAE:  {mae:.4f}°C")
    print(f"MedAE: {medae:.4f}°C")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²:   {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'medae': medae, 'mape': mape, 'r2': r2}

metrics_train = evaluate_predictions(y_train, y_pred_train, "Training")
metrics_val = evaluate_predictions(y_val, y_pred_val, "Validation")
metrics_test = evaluate_predictions(y_test, y_pred_test, "Test")

# ============ 10. LEARNING CURVES ============
print("\nGenerating learning curves...")

# Get evaluation history from model
eval_results = model.evals_result()

train_rmse = eval_results['validation_0']['rmse']
val_rmse = eval_results['validation_1']['rmse']

plt.figure(figsize=(12, 6))
epochs = range(1, len(train_rmse) + 1)
plt.plot(epochs, train_rmse, label='Training RMSE', linewidth=2)
plt.plot(epochs, val_rmse, label='Validation RMSE', linewidth=2)
plt.axvline(x=model.best_iteration, color='r', linestyle='--', 
            label=f'Best Iteration: {model.best_iteration}')
plt.xlabel('Boosting Round')
plt.ylabel('RMSE (°C)')
plt.title('XGBoost Learning Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('xgboost_learning_curves.png', dpi=300)
plt.show()

# ============ 11. VISUALIZATIONS ============

# Plot 1: Actual vs Predicted (All sets)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

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
plt.savefig('xgboost_predictions.png', dpi=300)
plt.show()

# Plot 2: Residuals Analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (y_true, y_pred, name, _) in zip(axes, datasets):
    residuals = y_true - y_pred
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Residuals (°C)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{name} Set Residuals\nMean = {residuals.mean():.4f}°C')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xgboost_residuals.png', dpi=300)
plt.show()

# Plot 3: Residuals vs Predicted (Test Set)
plt.figure(figsize=(10, 6))
residuals_test = y_test - y_pred_test
plt.scatter(y_pred_test, residuals_test, alpha=0.5, s=20)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted LST (°C)')
plt.ylabel('Residuals (°C)')
plt.title('Test Set: Residuals vs Predicted Values')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('xgboost_residuals_vs_predicted.png', dpi=300)
plt.show()

# Plot 4: Residuals vs Time (Test Set)
plt.figure(figsize=(14, 6))
plt.scatter(range(len(residuals_test)), residuals_test, alpha=0.6, s=30)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Time Index (Test Set)')
plt.ylabel('Residuals (°C)')
plt.title('Test Set: Residuals vs Time (Checking for Temporal Drift)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('xgboost_residuals_vs_time.png', dpi=300)
plt.show()

# Plot 5: Time Series Predictions (Sample Ward)
sample_ward = df['ward'].iloc[0]
ward_mask = df['ward'] == sample_ward
ward_data = df[ward_mask].sort_values(['year', 'month'])
ward_indices = ward_mask.values

ward_pred = model.predict(X[ward_indices])

plt.figure(figsize=(14, 6))
time_idx = range(len(ward_data))
plt.plot(time_idx, y[ward_indices], label='Actual LST', alpha=0.8, linewidth=2)
plt.plot(time_idx, ward_pred, label='Predicted LST', alpha=0.8, linestyle='--', linewidth=2)
plt.xlabel('Time (Months)')
plt.ylabel('LST (°C)')
plt.title(f'Time Series: Actual vs Predicted LST - Ward {sample_ward}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('xgboost_timeseries_ward.png', dpi=300)
plt.show()

# Plot 6: Time Series - Test Period Only
plt.figure(figsize=(14, 6))
test_ward_mask = (df['ward'] == sample_ward) & test_mask
test_ward_data = df[test_ward_mask].sort_values(['year', 'month'])
test_ward_indices = test_ward_mask.values

if len(test_ward_data) > 0:
    test_ward_pred = model.predict(X[test_ward_indices])
    test_time_idx = range(len(test_ward_data))
    
    plt.plot(test_time_idx, y[test_ward_indices], label='Actual LST (Test)', 
             alpha=0.8, linewidth=2, marker='o')
    plt.plot(test_time_idx, test_ward_pred, label='Predicted LST (Test)', 
             alpha=0.8, linestyle='--', linewidth=2, marker='s')
    plt.xlabel('Time (Months - 2025)')
    plt.ylabel('LST (°C)')
    plt.title(f'Test Period (2025): Actual vs Predicted LST - Ward {sample_ward}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('xgboost_timeseries_test_period.png', dpi=300)
    plt.show()

# Plot 7: Q-Q plot for residuals (Test Set)
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
plt.savefig('xgboost_diagnostics.png', dpi=300)
plt.show()

# ============ 12. SHAP ANALYSIS ============
print("\nGenerating SHAP Analysis...")

try:
    import shap
    
    # Use a smaller sample for SHAP
    sample_size = min(300, len(X_test))  # Reduced since test set is smaller
    X_test_sample = X_test.sample(n=sample_size, random_state=42)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)
    
    # SHAP Summary Plot
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_columns, show=False)
    plt.title('SHAP Feature Importance Summary')
    plt.tight_layout()
    plt.savefig('xgboost_shap_summary.png', dpi=300)
    plt.show()
    
    # SHAP Dependence plots for top 5 features
    top_features = importance_gain['feature'].head(5).tolist()
    print(f"\nCreating SHAP dependence plots for top 5 features: {top_features}")
    
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X_test_sample, 
                           feature_names=feature_columns, show=False)
        plt.title(f'SHAP Dependence Plot: {feature}')
        plt.tight_layout()
        plt.savefig(f'xgboost_shap_dependence_{feature}.png', dpi=300)
        plt.show()
    
    # SHAP Waterfall Plot for one prediction
    print("\nCreating SHAP Waterfall Plot for a single prediction...")
    
    sample_idx = 0
    plt.figure(figsize=(14, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=explainer.expected_value,
            data=X_test_sample.iloc[sample_idx],
            feature_names=feature_columns
        ),
        show=False,
        max_display=15
    )
    plt.title(f'SHAP Waterfall Plot - Prediction {sample_idx}\nActual: {y_test.iloc[sample_idx]:.2f}°C, Predicted: {model.predict(X_test.iloc[[sample_idx]])[0]:.2f}°C')
    plt.tight_layout()
    plt.savefig('xgboost_shap_waterfall.png', dpi=300)
    plt.show()
    
    print("SHAP analysis complete!")
    
except Exception as e:
    print(f"SHAP analysis failed: {e}")
    print("Continuing without SHAP plots...")

# ============ 13. SAVE RESULTS ============

# Save predictions separately for each set
train_results = pd.DataFrame({
    'ward': df[train_mask]['ward'],
    'zone': df[train_mask]['zone'],
    'year': df[train_mask]['year'],
    'month': df[train_mask]['month'],
    'actual_lst': y_train,
    'predicted_lst': y_pred_train,
    'residual': y_train - y_pred_train
})
train_results.to_csv('xgboost_train_predictions.csv', index=False)

val_results = pd.DataFrame({
    'ward': df[val_mask]['ward'],
    'zone': df[val_mask]['zone'],
    'year': df[val_mask]['year'],
    'month': df[val_mask]['month'],
    'actual_lst': y_val,
    'predicted_lst': y_pred_val,
    'residual': y_val - y_pred_val
})
val_results.to_csv('xgboost_validation_predictions.csv', index=False)

test_results = pd.DataFrame({
    'ward': df[test_mask]['ward'],
    'zone': df[test_mask]['zone'],
    'year': df[test_mask]['year'],
    'month': df[test_mask]['month'],
    'actual_lst': y_test,
    'predicted_lst': y_pred_test,
    'residual': y_test - y_pred_test
})
test_results.to_csv('xgboost_test_predictions.csv', index=False)

# Save all predictions
all_results = pd.DataFrame({
    'ward': df['ward'],
    'zone': df['zone'],
    'year': df['year'],
    'month': df['month'],
    'actual_lst': y,
    'predicted_lst': all_predictions,
    'residual': y - all_predictions
})
all_results.to_csv('xgboost_predictions.csv', index=False)

print("\nPredictions saved to CSV files")

backend_model_path = Path(__file__).resolve().parent.parent.parent / "backend" / "model" / "xgboost_model.pkl"
backend_model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, backend_model_path)
print(f"Saved trained model to {backend_model_path}")

# Save feature importance
importance_gain.to_csv('xgboost_feature_importance_gain.csv', index=False)
perm_importance_df.to_csv('xgboost_permutation_importance.csv', index=False)
print("Feature importance saved")

# Save best hyperparameters
best_params_df = pd.DataFrame({
    'parameter': list(random_search.best_params_.keys()),
    'value': list(random_search.best_params_.values())
})
best_params_df.to_csv('xgboost_best_hyperparameters.csv', index=False)
print("Best hyperparameters saved")

# Save metrics
metrics_df = pd.DataFrame({
    'Dataset': ['Training', 'Validation', 'Test'],
    'RMSE': [metrics_train['rmse'], metrics_val['rmse'], metrics_test['rmse']],
    'MAE': [metrics_train['mae'], metrics_val['mae'], metrics_test['mae']],
    'MedAE': [metrics_train['medae'], metrics_val['medae'], metrics_test['medae']],
    'MAPE': [metrics_train['mape'], metrics_val['mape'], metrics_test['mape']],
    'R2': [metrics_train['r2'], metrics_val['r2'], metrics_test['r2']]
})
metrics_df.to_csv('xgboost_metrics.csv', index=False)
print("Metrics saved to 'xgboost_metrics.csv'")

# ============ 14. FINAL SUMMARY REPORT ============
print("\n" + "="*50)
print("XGBOOST MODEL SUMMARY REPORT")
print("="*50)
print(f"Model Type: XGBoost Regressor (Hyperparameter Tuned with TimeSeriesCV + Early Stopping)")
print(f"Number of Features: {len(feature_columns)}")
print(f"Training Samples: {len(X_train)}")
print(f"Validation Samples: {len(X_val)}")
print(f"Test Samples: {len(X_test)}")
print(f"Best Iteration (Early Stopping): {model.best_iteration}")
print(f"Best Validation Score: {model.best_score:.4f}")

print(f"\nBest Hyperparameters (from RandomizedSearchCV with TimeSeriesSplit):")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

print("\nTop 5 Most Important Features (Built-in - Gain):")
for idx, row in importance_gain.head(5).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:.4f}")

print("\nTop 5 Most Important Features (Permutation):")
for idx, row in perm_importance_df.head(5).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:.4f} (±{row['std']:.4f})")

print("\nTest Set Performance:")
print(f"RMSE: {metrics_test['rmse']:.4f}°C")
print(f"MAE:  {metrics_test['mae']:.4f}°C")
print(f"MedAE: {metrics_test['medae']:.4f}°C")
print(f"MAPE: {metrics_test['mape']:.2f}%")
print(f"R²:   {metrics_test['r2']:.4f}")

print("\n" + "="*50)
print("Model training and evaluation complete!")
print("="*50)

# ============ 15. LEAKAGE CHECK ============
print("\n" + "="*50)
print("DATA LEAKAGE CHECK")
print("="*50)

potentially_leaky = []
if 'lst_12month_avg' in feature_columns:
    potentially_leaky.append("lst_12month_avg (ensure it's computed from historical data)")
if 'ndvi_3month_avg' in feature_columns:
    potentially_leaky.append("ndvi_3month_avg (ensure it's computed from historical data)")
if 'lst_3month_avg' in feature_columns:
    potentially_leaky.append("lst_3month_avg (ensure it's computed from historical data)")

if potentially_leaky:
    print("\n⚠️  WARNING: The following features may leak target information:")
    for feat in potentially_leaky:
        print(f"  - {feat}")
    print("\nPlease verify these features can be computed BEFORE predicting the target.")
    print("If they require future information, they should be removed.")
else:
    print("\n✅ No obvious leaky features detected.")

print("="*50)