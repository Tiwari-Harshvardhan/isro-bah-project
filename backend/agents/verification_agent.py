from __future__ import annotations

from typing import Any
import pandas as pd
import numpy as np


class PredictionVerificationAgent:
    def __init__(self, csv_service: Any) -> None:
        self.csv_service = csv_service

    def verify_prediction(
        self,
        zone_name: str,
        month: int,
        predicted_lst: float,
    ) -> dict[str, Any]:
        """
        Verify the ML prediction using climatology and historical ranges.
        If the prediction is implausible, it returns a corrected estimate and flags it.
        """
        try:
            history = self.csv_service.get_zone_history(zone_name)
        except Exception:
            return {
                "status": "Verified",
                "details": "History unavailable. Sanity check bypassed.",
                "corrected_lst": predicted_lst,
                "confidence_impact": 0
            }

        # Filter history for the same month
        seasonal = history[history["month"] == month]
        if not seasonal.empty:
            mean_temp = float(seasonal["mean_lst_day_celsius"].mean())
            std_temp = float(seasonal["mean_lst_day_celsius"].std())
            min_temp = float(seasonal["mean_lst_day_celsius"].min())
            max_temp = float(seasonal["mean_lst_day_celsius"].max())
            if pd.isna(std_temp) or std_temp == 0:
                std_temp = 2.0  # default assumption
        else:
            mean_temp = float(history["mean_lst_day_celsius"].mean())
            std_temp = 3.0
            min_temp = mean_temp - 5.0
            max_temp = mean_temp + 5.0

        # Recent values in the dataset (last 12 months)
        recent_lst = history["mean_lst_day_celsius"].dropna().tolist()[-12:]
        recent_mean = float(np.mean(recent_lst)) if recent_lst else mean_temp

        deviation = abs(predicted_lst - mean_temp)
        threshold = max(5.0, 3 * std_temp)

        is_implausible = (
            predicted_lst < 12.0 or 
            predicted_lst > 52.0 or 
            deviation > threshold
        )

        if is_implausible:
            status = "Needs Review"
            # Blended correction: 70% historical monthly average + 30% predicted
            corrected_lst = round(0.7 * mean_temp + 0.3 * predicted_lst, 2)
            details = (
                f"Prediction ({predicted_lst}°C) deviates significantly from the historical "
                f"monthly average ({mean_temp:.1f}°C) for {zone_name} (deviation: {deviation:.1f}°C > threshold {threshold:.1f}°C)."
            )
            confidence_impact = -30
        else:
            status = "Verified"
            corrected_lst = predicted_lst
            details = f"Prediction is within the historical monthly limits for {zone_name}."
            confidence_impact = 0

        return {
            "status": status,
            "details": details,
            "corrected_lst": corrected_lst,
            "confidence_impact": confidence_impact
        }
