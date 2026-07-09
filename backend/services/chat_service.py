from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from backend.config import GEMINI_API_KEY
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

        prompt = self._build_prompt(zone, query, summary, prediction, analysis)
        ai_answer = self._query_gemini(prompt)

        if ai_answer:
            answer = ai_answer
        else:
            answer = self._fallback_answer(zone, query, summary, prediction, analysis, year, month)

        return {
            'query': query,
            'answer': answer,
            'prediction': prediction,
            'analysis': analysis,
        }

    def _build_prompt(self, zone: str, query: str, summary: dict[str, Any], prediction: dict[str, Any], analysis: dict[str, Any]) -> str:
        return (
            f"You are an intelligent urban heat assistant for Delhi zones. "
            f"Use the summary, prediction, and analysis to answer the user's question clearly and concisely. "
            f"Zone: {zone}. "
            f"Summary: population {summary.get('population')}, density {summary.get('population_density')}, "
            f"built-up {summary.get('built_up_percent')}%, NDVI {summary.get('mean_ndvi')}. "
            f"Prediction: {prediction.get('predicted_lst')}°C with confidence {analysis.get('confidence')}%. "
            f"Reasoning: {'; '.join(analysis.get('reasoning', []))}. "
            f"Historical comparison: {analysis.get('historical_comparison')}. "
            f"Neighbour comparison: {analysis.get('neighbour_comparison')}. "
            f"User question: {query}"
        )

    def _query_gemini(self, prompt: str) -> str | None:
        if not GEMINI_API_KEY:
            return None

        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Authorization': f'Bearer {GEMINI_API_KEY}',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': 'gpt-4.1-mini',
            'messages': [
                {'role': 'system', 'content': 'You are a concise urban heat intelligence assistant for Delhi zones.'},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': 0.6,
            'max_tokens': 400,
        }

        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method='POST',
            )
            with urllib.request.urlopen(request, timeout=25) as response:
                body = json.loads(response.read().decode('utf-8'))
                choices = body.get('choices', [])
                if choices:
                    message = choices[0].get('message', {})
                    content = message.get('content')
                    if isinstance(content, str):
                        return content.strip()
                    if isinstance(content, dict):
                        return content.get('text', '').strip()
        except urllib.error.HTTPError:
            return None
        except Exception:
            return None

        return None

    def _fallback_answer(self, zone: str, query: str, summary: dict[str, Any], prediction: dict[str, Any], analysis: dict[str, Any], year: int, month: int) -> str:
        query_lower = query.strip().lower()

        if 'hottest' in query_lower:
            hottest_zone = self._find_hottest_zone(year, month)
            return f"The hottest zone in this dataset for {month}/{year} is {hottest_zone['zone']} with {hottest_zone['predicted_lst']:.1f}°C."
        if 'compare' in query_lower:
            return self._compare_zones(query_lower, year, month)
        if 'lowest ndvi' in query_lower:
            lowest = self._zone_with_lowest_ndvi()
            return f"{lowest['zone']} has the lowest NDVI at {lowest['mean_ndvi']:.2f}, indicating the greatest vegetation deficit."
        if 'suggest mitigation' in query_lower or 'mitigation' in query_lower:
            return self._recommend_mitigation(zone, prediction, summary)
        if 'explain' in query_lower or 'reason' in query_lower or 'why' in query_lower:
            return self._explain_prediction(analysis)

        return f"Prediction for {zone} is {prediction['predicted_lst']:.1f}°C with {analysis['confidence']}% confidence. {self._explain_prediction(analysis)}"

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
