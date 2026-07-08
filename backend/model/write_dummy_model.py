import joblib
from pathlib import Path

from backend.model.dummy_model import DummyModel

path = Path(__file__).resolve().parent / "xgboost_model.pkl"
joblib.dump(DummyModel(), path)
print(f"saved dummy model to {path}")
