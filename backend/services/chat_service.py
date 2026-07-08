from __future__ import annotations

from typing import Any

from backend.services.analysis_service import AIAnalysisService
from backend.services.csv_service import CSVService
from backend.services.prediction_service import PredictionService


class AIChatService:
    def __init__(self) -> None:
        self.analysis_service = AIAnalysisService()
        self.csv_service = CSVService()
        self.prediction_service = PredictionService()

    def answer_query(self, zone: str, year: int, month: int, query: str) -> dict[str, Any]:
        summary = self.csv_service.get_zone_summary(zone)
        prediction = self.prediction_service.predict_for_zone(zone, year, month)
        analysis = self.analysis_service.explain_prediction(zone, year, month)
        query_lower = query.strip().lower()

        if 'hottest' in query_lower:
            hottest_zone = self._find_hottest_zone(year, month)
            answer = f"The hottest zone for {month}/{year} is {hottest_zone['zone']} with {hottest_zone['predicted_lst']:.1f}°C."
        elif 'compare' in query_lower:
            answer = self._compare_zones(query_lower, year, month)
        elif 'lowest ndvi' in query_lower:
            lowest = self._zone_with_lowest_ndvi()
            answer = f"{lowest['zone']} has the lowest NDVI at {lowest['mean_ndvi']:.2f}, indicating the greatest vegetation deficit."
        elif 'suggest mitigation' in query_lower or 'mitigation' in query_lower:
            answer = self._recommend_mitigation(zone, prediction, summary)
        elif 'explain' in query_lower or 'reason' in query_lower or 'why' in query_lower:
            answer = self._explain_prediction(analysis)
        else:
            answer = f"Prediction for {zone} is {prediction['predicted_lst']:.1f}°C with {analysis['confidence']}% confidence. {self._explain_prediction(analysis)}"

        return {
            'query': query,
            'answer': answer,
            'prediction': prediction,
            'analysis': analysis,
        }

    def _find_hottest_zone(self, year: int, month: int) -> dict[str, Any]:
        dataset = self.csv_service.load_dataset()
        latest = dataset[(dataset['year'] == year) & (dataset['month'] == month)]
        best = latest.sort_values('mean_lst_day_celsius', ascending=False).iloc[0]
        return {'zone': best['zone'], 'predicted_lst': best['mean_lst_day_celsius']}

    def _compare_zones(self, query: str, year: int, month: int) -> str:
        names = [token.strip() for token in query.replace('compare', '').split('and') if token.strip()]
        if len(names) < 2:
            return 'Please specify two zones to compare.'
        comparisons = []
        for name in names[:2]:
            try:
                summary = self.csv_service.get_zone_summary(name)
                prediction = self.prediction_service.predict_for_zone(name, year, month)
                comparisons.append(f"{name}: {prediction['predicted_lst']:.1f}°C, NDVI {summary['mean_ndvi']:.2f}, built-up {summary['built_up_percent']:.1f}%.")
            except Exception:
                comparisons.append(f"{name}: data unavailable.")
        return ' '.join(comparisons)

    def _zone_with_lowest_ndvi(self) -> dict[str, Any]:
        dataset = self.csv_service.load_dataset()
        latest = dataset.sort_values(['year', 'month']).groupby('zone').tail(1)
        lowest = latest.loc[latest['mean_ndvi'].idxmin()]
        return lowest.to_dict()

    def _recommend_mitigation(self, zone: str, prediction: dict[str, Any], summary: dict[str, Any]) -> str:
        advice = []
        if summary['mean_ndvi'] < 0.25:
            advice.append('Increase green cover and tree planting.')
        if summary['built_up_percent'] > 50:
            advice.append('Implement cool roofing and reflective pavement.')
        if summary['population_density'] > 25000:
            advice.append('Prioritize shaded public spaces and water features.')
        return ' '.join(advice) if advice else 'Continue monitoring and maintain current mitigation strategies.'

    def _explain_prediction(self, analysis: dict[str, Any]) -> str:
        return ' '.join(analysis.get('reasoning', []))
