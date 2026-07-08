from __future__ import annotations

from typing import Any

from backend.model.model_loader import load_model
from backend.services.csv_service import CSVService
from backend.utils.feature_engineering import engineer_features


class PredictionService:
    def __init__(self) -> None:
        self.csv_service = CSVService()
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.model = load_model()
        except FileNotFoundError:
            self.model = None

    def predict_for_zone(self, zone_name: str, year: int | None = None, month: int | None = None) -> dict[str, Any]:
        dataset = self.csv_service.load_dataset()
        zone_rows = dataset[dataset["zone"].astype(str).str.lower() == zone_name.lower()].copy()
        if zone_rows.empty:
            raise ValueError(f"No data found for zone {zone_name}")

        zone_rows = zone_rows.sort_values(["year", "month"]).reset_index(drop=True)
        if year is not None and month is not None:
            filtered = zone_rows[(zone_rows["year"] == year) & (zone_rows["month"] == month)]
            if filtered.empty:
                filtered = zone_rows.iloc[[-1]]
            selected = filtered.iloc[0]
        else:
            selected = zone_rows.iloc[-1]

        feature_frame = zone_rows.copy()
        feature_frame = engineer_features(feature_frame)
        if year is not None and month is not None:
            candidate = feature_frame[(zone_rows["year"] == year) & (zone_rows["month"] == month)]
            if candidate.empty:
                candidate = feature_frame.iloc[[-1]]
            feature_row = candidate.iloc[0:1]
        else:
            feature_row = feature_frame.iloc[[-1]]

        if self.model is None:
            prediction = float(selected.get("mean_lst_day_celsius", 0.0))
        else:
            prediction = float(self.model.predict(feature_row)[0])
        historical_lst = [float(value) for value in zone_rows["mean_lst_day_celsius"].dropna().tolist()]
        risk_level = self._risk_level(prediction)
        recommendations = self._recommendations(prediction, float(selected.get("built_up_percent", 0.0)), float(selected.get("mean_ndvi", 0.0)), float(selected.get("population_density", 0.0)))

        return {
            "zone": zone_name,
            "year": int(selected.get("year", year or 0)),
            "month": int(selected.get("month", month or 0)),
            "predicted_lst": round(prediction, 2),
            "historical_lst": round(float(selected.get("mean_lst_day_celsius", historical_lst[-1] if historical_lst else 0.0)), 2),
            "population": float(selected.get("population", 0.0)),
            "population_density": float(selected.get("population_density", 0.0)),
            "built_up_percent": float(selected.get("built_up_percent", 0.0)),
            "mean_ndvi": float(selected.get("mean_ndvi", 0.0)),
            "risk_level": risk_level,
            "recommendation": recommendations,
        }

    def _risk_level(self, predicted_temp: float) -> str:
        if predicted_temp < 32:
            return "Low"
        if predicted_temp < 36:
            return "Moderate"
        if predicted_temp < 40:
            return "High"
        return "Extreme"

    def _recommendations(self, predicted_temp: float, built_up_percent: float, mean_ndvi: float, population_density: float) -> list[str]:
        recommendations = []
        if mean_ndvi < 0.2:
            recommendations.append("High NDVI missing → Plant trees")
        if built_up_percent > 50:
            recommendations.append("High built-up → Cool roofs")
        if population_density > 25000:
            recommendations.append("High density → Increase green spaces")
        if predicted_temp >= 40:
            recommendations.append("Very high temperature → Immediate heat mitigation")
        if not recommendations:
            recommendations = ["Keep current cooling and greening measures in place."]
        return recommendations[:5]
