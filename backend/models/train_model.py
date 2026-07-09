import joblib
import pandas as pd
from xgboost import XGBRegressor
from backend.config import DATASET_PATH, MODEL_PATH
from backend.utils.feature_engineering import engineer_features, FEATURE_COLUMNS

def train() -> None:
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    
    # Normalize string columns to prevent nan type issues
    for col in ["zone", "ward"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
            
    # Drop rows where target mean_lst_day_celsius is NaN
    df_clean = df.dropna(subset=["mean_lst_day_celsius"]).copy()
    
    print("Engineering features using feature_engineering module...")
    X = engineer_features(df_clean)
    y = df_clean["mean_lst_day_celsius"]
    
    print(f"Training data size: {X.shape[0]} rows, {X.shape[1]} columns")
    
    # Train XGBoost regressor
    print("Training XGBRegressor...")
    model = XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    print(f"Saving trained model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("XGBoost model training completed successfully!")

if __name__ == "__main__":
    train()
