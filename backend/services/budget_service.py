from __future__ import annotations

from typing import Any
import pandas as pd

from backend.services.csv_service import CSVService
from backend.services.prediction_service import PredictionService


class BudgetPlannerService:
    def __init__(self) -> None:
        self.csv_service = CSVService()
        self.prediction_service = PredictionService()

    def plan_budget(self, budget: float, year: int, month: int) -> dict[str, Any]:
        dataset = self.csv_service.load_dataset()
        
        # Group by zone and take the latest record for each zone to get latest stats
        latest = dataset.sort_values(['year', 'month']).groupby('zone').tail(1).copy()
        latest = latest.dropna(subset=['zone'])

        # Compute risk score considering LST, density, built-up, and NDVI
        latest['risk_score'] = (
            (latest['mean_lst_day_celsius'] / 45.0) * 0.35 +
            (latest['population_density'] / 30000.0) * 0.25 +
            (latest['built_up_percent'] / 100.0) * 0.25 +
            ((1.0 - latest['mean_ndvi'].fillna(0.2)) / 1.0) * 0.15
        )
        
        # Priority score integrates the risk and total population to be cooled
        latest['priority_score'] = latest['risk_score'] * latest['population']
        latest = latest.sort_values('priority_score', ascending=False)

        total_priority = latest['priority_score'].sum()
        allocations = []
        for _, row in latest.iterrows():
            share = float(row['priority_score'] / total_priority) if total_priority else 0
            amount = round(budget * share, 2)
            allocations.append({
                'zone': row['zone'],
                'suggested_budget': amount,
                'expected_population': int(row['population']),
                'priority_score': float(row['priority_score']),
                'risk_level': self._risk_level(float(row['mean_lst_day_celsius'])),
                'suggested_intervention': self._choose_intervention(row),
            })

        return {
            'budget': budget,
            'estimated_population_benefited': int(latest['population'].sum()),
            'estimated_cost_efficiency': round(budget / (latest['population'].sum() or 1), 5),
            'priority_summary': allocations,
        }

    def _risk_level(self, predicted_temp: float) -> str:
        if predicted_temp < 32:
            return 'Low'
        if predicted_temp < 36:
            return 'Moderate'
        if predicted_temp < 40:
            return 'High'
        return 'Extreme'

    def _choose_intervention(self, row: pd.Series) -> str:
        interventions = []
        if row['mean_ndvi'] < 0.25:
            interventions.append('Plant trees and green corridors')
        if row['built_up_percent'] > 50:
            interventions.append('Deploy cool roofs and reflective pavements')
        if row['population_density'] > 25000:
            interventions.append('Create shaded public spaces and water features')
        return '; '.join(interventions) if interventions else 'Maintain current heat mitigation programs'
