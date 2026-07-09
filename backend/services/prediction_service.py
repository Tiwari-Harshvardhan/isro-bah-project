from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from backend.models.model_loader import load_model
from backend.services.csv_service import CSVService
from backend.utils.feature_engineering import engineer_features
from backend.agents.verification_agent import PredictionVerificationAgent


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
        
        # Determine target year and month
        if year is not None and month is not None:
            filtered = zone_rows[(zone_rows["year"] == year) & (zone_rows["month"] == month)]
            if filtered.empty:
                # Fallback to latest available year/month in dataset for this zone
                latest_year = int(zone_rows["year"].max())
                latest_month = int(zone_rows[zone_rows["year"] == latest_year]["month"].max())
                filtered = zone_rows[(zone_rows["year"] == latest_year) & (zone_rows["month"] == latest_month)]
        else:
            latest_year = int(zone_rows["year"].max())
            latest_month = int(zone_rows[zone_rows["year"] == latest_year]["month"].max())
            filtered = zone_rows[(zone_rows["year"] == latest_year) & (zone_rows["month"] == latest_month)]

        # Get all ward rows corresponding to selected year and month
        selected_year = int(filtered.iloc[0]["year"])
        selected_month = int(filtered.iloc[0]["month"])
        
        target_rows = zone_rows[(zone_rows["year"] == selected_year) & (zone_rows["month"] == selected_month)].copy()
        
        # Run feature engineering on the full zone set to preserve lags
        feature_frame = zone_rows.copy()
        feature_frame = engineer_features(feature_frame)
        
        # Filter feature frame for target indices
        feature_rows = feature_frame.loc[target_rows.index]

        # Inferences
        if self.model is None:
            prediction = float(target_rows["mean_lst_day_celsius"].mean())
        else:
            predictions = self.model.predict(feature_rows)
            # Remove any NaNs if present
            valid_predictions = predictions[~np.isnan(predictions)]
            if len(valid_predictions) > 0:
                prediction = float(np.mean(valid_predictions))
            else:
                prediction = float(target_rows["mean_lst_day_celsius"].mean())

        historical_lst_series = zone_rows["mean_lst_day_celsius"].dropna().tolist()
        historical_mean = float(np.mean(historical_lst_series)) if historical_lst_series else 30.0
        historical_max = float(np.max(historical_lst_series)) if historical_lst_series else 40.0
        
        # Sanity check with prediction verification agent
        verification_agent = PredictionVerificationAgent(self.csv_service)
        verification = verification_agent.verify_prediction(zone_name, selected_month, prediction)
        
        status = verification["status"]
        details = verification["details"]
        corrected_lst = verification["corrected_lst"]
        confidence_impact = verification["confidence_impact"]

        # Calculate final confidence
        base_confidence = self._estimate_confidence(prediction, historical_mean, historical_max)
        confidence = max(10, min(95, base_confidence + confidence_impact))

        risk_level = self._risk_level(corrected_lst)
        
        # Aggregate zone stats
        total_population = float(target_rows["population"].sum())
        mean_density = float(target_rows["population_density"].mean())
        mean_built_up = float(target_rows["built_up_percent"].mean())
        mean_ndvi_val = float(target_rows["mean_ndvi"].mean())

        recommendations = self._recommendations(
            corrected_lst,
            mean_built_up,
            mean_ndvi_val,
            mean_density,
            total_population
        )

        return {
            "zone": zone_name,
            "year": selected_year,
            "month": selected_month,
            "predicted_lst": round(corrected_lst, 2),
            "historical_lst": round(float(target_rows["mean_lst_day_celsius"].mean() or (historical_lst_series[-1] if historical_lst_series else 0.0)), 2),
            "population": total_population,
            "population_density": mean_density,
            "built_up_percent": mean_built_up,
            "mean_ndvi": mean_ndvi_val,
            "risk_level": risk_level,
            "recommendation": recommendations,
            "verification_status": status,
            "verification_details": details,
            "original_predicted_lst": round(prediction, 2),
            "confidence": confidence
        }

    def _risk_level(self, predicted_temp: float) -> str:
        if predicted_temp < 32:
            return "Low"
        if predicted_temp < 36:
            return "Moderate"
        if predicted_temp < 40:
            return "High"
        return "Extreme"

    def _estimate_confidence(self, predicted: float, historical_mean: float, historical_max: float) -> int:
        score = 90
        if historical_max > 0 and predicted > historical_max + 3:
            score -= 30
        if abs(predicted - historical_mean) > 3:
            score -= 20
        if predicted < 30 or predicted > 45:
            score -= 10
        return max(30, min(95, score))

    def _recommendations(self, predicted_temp: float, built_up_percent: float, mean_ndvi: float, population_density: float, population: float) -> list[dict[str, Any]]:
        recommendations = []
        if mean_ndvi < 0.25:
            recommendations.append({
                "title": "Urban Afforestation & Green Corridors",
                "description": "Plant native trees and create continuous green pathways to offset low NDVI vegetation cover.",
                "expected_cooling_impact": "1.5°C - 2.5°C reduction",
                "estimated_implementation_cost": "₹75 Lakhs",
                "priority": "High",
                "affected_population": int(population * 0.6)
            })
        if built_up_percent > 50:
            recommendations.append({
                "title": "Cool Roof Initiative",
                "description": "Apply high-albedo reflective coatings to residential and commercial rooftops to reduce thermal absorption.",
                "expected_cooling_impact": "1.0°C - 2.0°C reduction",
                "estimated_implementation_cost": "₹1.2 Crore",
                "priority": "High",
                "affected_population": int(population * 0.8)
            })
        if population_density > 20000:
            recommendations.append({
                "title": "Community Pocket Parks & Shaded Public Spaces",
                "description": "Establish small-scale green spaces and water misters in high-density residential blocks.",
                "expected_cooling_impact": "0.8°C - 1.5°C reduction",
                "estimated_implementation_cost": "₹50 Lakhs",
                "priority": "Medium",
                "affected_population": int(population * 0.5)
            })
        if predicted_temp >= 38.0:
            recommendations.append({
                "title": "Reflective Pavement & Cool Streets",
                "description": "Install reflective asphalt or light-colored pavers on streets and parking lots to curb heat retention.",
                "expected_cooling_impact": "1.2°C - 2.2°C reduction",
                "estimated_implementation_cost": "₹1.5 Crore",
                "priority": "High" if predicted_temp < 41.0 else "Extreme",
                "affected_population": int(population)
            })
            
        if not recommendations:
            recommendations.append({
                "title": "Ongoing Heat Canopy Maintenance",
                "description": "Keep current cooling and greening measures in place. Monitor NDVI stability annually.",
                "expected_cooling_impact": "0.2°C - 0.5°C maintenance",
                "estimated_implementation_cost": "₹10 Lakhs",
                "priority": "Low",
                "affected_population": int(population * 0.3)
            })
            
        return recommendations
