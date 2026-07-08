from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import DATASET_PATH


class CSVService:
    def __init__(self, dataset_path: Path | None = None) -> None:
        self.dataset_path = dataset_path or DATASET_PATH

    def load_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataset_path)
        return self._normalize_dataset(df)

    def get_zone_history(self, zone_name: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
        data = self.load_dataset() if df is None else df
        zone_history = data[data["zone"].astype(str).str.lower() == zone_name.lower()].copy()
        if zone_history.empty:
            raise ValueError(f"No data found for zone {zone_name}")
        zone_history = zone_history.sort_values(["year", "month"]).reset_index(drop=True)
        return zone_history

    def get_zone_summary(self, zone_name: str) -> dict[str, Any]:
        zone_history = self.get_zone_history(zone_name)
        latest = zone_history.iloc[-1]
        return {
            "zone": zone_name,
            "number_of_wards": int(zone_history["ward"].nunique()),
            "population": float(latest.get("population", 0.0)),
            "population_density": float(latest.get("population_density", 0.0)),
            "built_up_percent": float(latest.get("built_up_percent", 0.0)),
            "mean_ndvi": float(latest.get("mean_ndvi", 0.0)),
            "historical_lst": [float(value) for value in zone_history["mean_lst_day_celsius"].dropna().tolist()],
            "latest_year": int(latest.get("year", 0)),
            "latest_month": int(latest.get("month", 0)),
        }

    def _normalize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        for column in ["zone", "ward"]:
            if column in normalized.columns:
                normalized[column] = normalized[column].fillna("").astype(str)
        for column in ["year", "month"]:
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        return normalized
