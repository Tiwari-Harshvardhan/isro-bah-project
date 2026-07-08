from __future__ import annotations

from typing import Any

import pandas as pd

from backend.services.csv_service import CSVService
from backend.services.prediction_service import PredictionService


class AIAnalysisService:
    def __init__(self) -> None:
        self.csv_service = CSVService()
        self.prediction_service = PredictionService()

    def explain_prediction(self, zone_name: str, year: int, month: int) -> dict[str, Any]:
        summary = self.csv_service.get_zone_summary(zone_name)
        prediction = self.prediction_service.predict_for_zone(zone_name, year, month)
        history = self.csv_service.get_zone_history(zone_name)

        historical_mean = float(history["mean_lst_day_celsius"].mean()) if not history.empty else 0.0
        historical_max = float(history["mean_lst_day_celsius"].max()) if not history.empty else 0.0
        predicted = float(prediction["predicted_lst"])
        confidence = self._estimate_confidence(predicted, historical_mean, historical_max)

        reasoning = [
            f"Historical average LST for {zone_name} is {historical_mean:.1f}°C.",
            f"Built-up percentage is {summary['built_up_percent']:.1f}%.",
            f"NDVI is {summary['mean_ndvi']:.2f}.",
            f"Population density is {summary['population_density']:.0f} people/km².",
        ]
        if predicted > historical_max:
            reasoning.append("Prediction is above the historical maximum for this zone.")

        seasonal = self._seasonal_comparison(history, year, month)
        neighbour = self._neighbour_comparison(zone_name, predicted)
        anomalies = self._anomaly_check(predicted, historical_mean)

        return {
            "prediction": predicted,
            "confidence": confidence,
            "reasoning": reasoning,
            "historical_comparison": seasonal,
            "neighbour_comparison": neighbour,
            "potential_anomalies": anomalies,
        }

    def _estimate_confidence(self, predicted: float, historical_mean: float, historical_max: float) -> int:
        score = 90
        if historical_max > 0 and predicted > historical_max + 3:
            score -= 30
        if abs(predicted - historical_mean) > 3:
            score -= 20
        if predicted < 30 or predicted > 45:
            score -= 10
        return max(30, min(95, score))

    def _seasonal_comparison(self, history: pd.DataFrame, year: int, month: int) -> str:
        seasonal = history[(history["month"] == month) & (history["year"] != year)]
        if seasonal.empty:
            return "No previous seasonal history available."
        seasonal_mean = float(seasonal["mean_lst_day_celsius"].mean())
        return f"Average for month {month} in past years is {seasonal_mean:.1f}°C."

    def _neighbour_comparison(self, zone_name: str, predicted: float) -> str:
        dataset = self.csv_service.load_dataset()
        latest = dataset.sort_values(["year", "month"]).groupby("zone").tail(1)
        neighbours = latest[latest["zone"] != zone_name]
        top = neighbours.sort_values("mean_lst_day_celsius", ascending=False).head(3)
        comparison = [f"{row['zone']} at {row['mean_lst_day_celsius']:.1f}°C" for _, row in top.iterrows()]
        return f"Nearby zone temperatures: {', '.join(comparison)}."

    def _anomaly_check(self, predicted: float, historical_mean: float) -> list[str]:
        issues: list[str] = []
        if abs(predicted - historical_mean) > 4:
            issues.append("Prediction differs significantly from recent history.")
        if predicted > 42:
            issues.append("Prediction is very high and should be manually verified.")
        return issues or ["No major anomalies detected."]
