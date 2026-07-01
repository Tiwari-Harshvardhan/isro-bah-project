import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../data/final/final_mp_dataset.csv')

feature_columns = [
    'year',
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

print(df.isna().sum())
df = df.dropna(subset=feature_columns + ['target_lst']).reset_index(drop=True)
print(df.shape)

X = df[feature_columns]
y = df['target_lst']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print("Feature names: ")
print(X.columns.tolist())

train_mask = df['year'].between(2018,2023)
val_mask = df['year'] == 2024
test_mask = df['year'] == 2025

X_train = X[train_mask]
y_train = y[train_mask]
X_val = X[val_mask]
y_val = y[val_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print("Data Split Summary: ")
print(f"Training set: {X_train.shape[0]} samples (2018-2023)")
print(f"Validation set: {X_val.shape[0]} samples (2024)")
print(f"Test set: {X_test.shape[0]} samples (2025)")

print("Training LightGBM Model...")

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'random_state': 42,
    'n_jobs': -1
}

#create lightgbm datasets
train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_columns)
val_data = lgb.Dataset(X_val, label = y_val, reference=train_data)

print("Training model...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets = [train_data, val_data],
    valid_names = ['train','validation'],
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]
)

print(f"best iteration: {model.best_iteration}")

#plot feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importance(importance_type = 'gain')
}).sort_values('importance', ascending=False)

print("Feature Importance")
print(feature_importance)

plt.figure(figsize = (10,6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance (Gain)')
plt.title('Light GBM Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

print("Making predictions")
y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
y_pred_val = model.predict(X_val, num_iteration = model.best_iteration)
y_pred_test = model.predict(X_test, num_iteration = model.best_iteration)
test_results = df.loc[test_mask].copy()
test_results['predicted_lst'] = y_pred_test
sample_district = test_results['district'].iloc[0]
district_data = test_results[test_results['district'] == sample_district].copy()
district_data['date'] = pd.to_datetime(dict(year=district_data.year, month=district_data.month, day=1))

plt.figure(figsize=(14,6))
plt.plot(district_data['date'], district_data['target_lst'], label='Actual LST')
plt.plot(district_data['date'], district_data['predicted_lst'], '--', label="Predicted LST")
plt.xlabel('Date')
plt.ylabel('LST(degree celcius)')
plt.title(f'Actual vs Predicted LST -{sample_district}')

plt.legend()
plt.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('time_series_predictions.png', dpi=300)
plt.show()

#calculate metrics
def evaluate_predictions(y_true, y_pred, dataset_name):
    """calculate and print evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{dataset_name} Metrics")
    print(f"RMSE: {rmse: .4f}C")
    print(f"MAE: {mae:.4f}C")
    print(f"R^2: {r2:.4f}")
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

metrics_train = evaluate_predictions(y_train, y_pred_train, "Training")
metrics_val = evaluate_predictions(y_val, y_pred_val, "Validation")
metrics_test = evaluate_predictions(y_test, y_pred_test, "Test")

#visualizations
#plot1: actual vs predicted
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(y_test, y_pred_test, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--',lw=2)
plt.xlabel('Actual LST (degree celcius)')
plt.ylabel('Predicted LST (degree celcius)')
plt.title(f'Test Set: Actual vs Predicted \n R^2 = {metrics_test["r2"]:.4f}')
plt.grid(True, alpha = 0.3)

plt.subplot(1,2,2)
residuals = y_test - y_pred_test
plt.hist(residuals, bins = 30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals (degree celcius)')
plt.ylabel('Frequency')
plt.title(f'Test Set: Residual Distribution\n Mean = {residuals.mean():.4f} degree celcius')
plt.axvline(x=0, color='r', linestyle = '--', lw=2)
plt.grid(True, alpha = 0.3)

plt.tight_layout()
plt.savefig('predictions_analysis.png', dpi=300)
plt.show()

#plot3: feature importance (horizontal bar)
plt.figure(figsize=(10,8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel("Importance (Gain)")
plt.title('LightGBM Feature Importance')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_detailed.png',dpi=300)
plt.show()

feature_importance.to_csv('feature_importance.csv', index=False)

#save model and results
model.save_model('urban_cool_lgb_model.txt')
print("Model saved as 'urban_cool_lgb_model.txt'")

results_df = pd.DataFrame({
    'district_code': df.loc[test_mask, 'district_code'].values,
    'district': df.loc[test_mask, 'district'].values,
    'year': df.loc[test_mask, 'year'].values,
    'month': df.loc[test_mask, 'month'].values,
    'actual_lst': y_test.values,
    'predicted_lst': y_pred_test,
    'error': y_test.values - y_pred_test
})

results_df.to_csv('predictions_results.csv', index=False)
print("predictions saved to 'predictions_results.csv'")

print("MODEL SUMMARY REPORT")
print("Model Type: LightGBM Regressor")
print(f"Number of features: {len(feature_columns)}")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")
print(f"\nBest Iteration: {model.best_iteration}")
print(f"Learning Rate: {params['learning_rate']}")
print(f"Num Leaves: {params['num_leaves']}")
print("\nTest Set Performance:")
print(f"RMSE: {metrics_test['rmse']:.4f}°C")
print(f"MAE:  {metrics_test['mae']:.4f}°C")
print(f"R²:   {metrics_test['r2']:.4f}")
print("\nTop 5 Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.2f}")


