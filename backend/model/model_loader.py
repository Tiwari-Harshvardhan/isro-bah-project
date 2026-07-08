from __future__ import annotations

import joblib

from backend.config import MODEL_PATH


def load_model():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)
