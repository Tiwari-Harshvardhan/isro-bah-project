from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any


class PlanningAssistantAgent:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def answer_question(
        self,
        zone: str,
        prediction: dict[str, Any],
        historical_data: list[dict[str, Any]],
        zone_statistics: dict[str, Any],
        question: str
    ) -> str:
        # Build prompt with exact statistics, predictions, and guidelines
        prompt = self._build_prompt(zone, prediction, historical_data, zone_statistics, question)
        
        if self.api_key:
            response_text = self._query_gemini(prompt)
            if response_text:
                return response_text

        # Fallback system if API is unavailable/offline
        return self._generate_fallback_response(zone, prediction, historical_data, zone_statistics, question)

    def _build_prompt(self, zone: str, prediction: dict[str, Any], historical_data: list[dict[str, Any]], zone_statistics: dict[str, Any], question: str) -> str:
        # Extract last few temperatures from history
        recent_temps = [float(row.get("mean_lst_day_celsius", 0.0)) for row in historical_data[-5:] if row.get("mean_lst_day_celsius")]
        recent_temps_str = ", ".join(f"{t:.1f}°C" for t in recent_temps)

        return (
            f"You are a professional urban heat intelligence planning AI assistant for the Delhi MCD dashboard.\n"
            f"Use the following real data to answer the policymaker's question. Do NOT make up any numbers. If the query asks for interventions or explanations, explain using the specific statistics below.\n\n"
            f"Zone: {zone}\n"
            f"Current Prediction: {prediction.get('predicted_lst', 0.0)}°C (Risk: {prediction.get('risk_level', 'Unknown')})\n"
            f"Verification Status: {prediction.get('verification_status', 'N/A')}\n"
            f"Verification Details: {prediction.get('verification_details', 'N/A')}\n"
            f"Zone Statistics: Population = {int(zone_statistics.get('population', 0)):,}, "
            f"Population Density = {int(zone_statistics.get('population_density', 0)):,} people/km², "
            f"Built-up Area = {zone_statistics.get('built_up_percent', 0.0):.1f}%, "
            f"NDVI (Vegetation Index) = {zone_statistics.get('mean_ndvi', 0.0):.3f}\n"
            f"Recent LST History: {recent_temps_str}\n\n"
            f"Policymaker Question: {question}\n\n"
            f"Instructions:\n"
            f"1. Rely only on the provided data. Do not hallucinate external metrics.\n"
            f"2. Answer in a professional, concise, policy-oriented format (2-4 sentences max).\n"
            f"3. Explain the relationship between low NDVI/high built-up and heat if asked.\n"
            f"4. If asked about trees, suggest tree counts based on density and NDVI (e.g., planting 10,000 to 50,000 trees to raise NDVI index by 0.1, benefiting the population).\n"
            f"5. If asked about budget allocation, mention how the allocation cools the residents."
        )

    def _query_gemini(self, prompt: str) -> str | None:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=20) as response:
                body = json.loads(response.read().decode("utf-8"))
                candidates = body.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return str(parts[0].get("text", "")).strip()
        except Exception:
            # Fall back to templated responses on any API error
            return None
        return None

    def _generate_fallback_response(
        self,
        zone: str,
        prediction: dict[str, Any],
        historical_data: list[dict[str, Any]],
        zone_statistics: dict[str, Any],
        question: str
    ) -> str:
        q = question.lower()
        predicted_lst = prediction.get("predicted_lst", 34.0)
        built_up = zone_statistics.get("built_up_percent", 50.0)
        mean_ndvi = zone_statistics.get("mean_ndvi", 0.2)
        density = zone_statistics.get("population_density", 20000)
        pop = zone_statistics.get("population", 100000)

        if "cause" in q or "why" in q or "reason" in q:
            return (
                f"The high land surface temperature of {predicted_lst:.2f}°C in {zone} is primarily driven by "
                f"its high built-up area ratio of {built_up:.1f}% and deficient vegetation index (NDVI: {mean_ndvi:.3f}). "
                f"With a population density of {int(density):,} people/km², anthropogenic heat and impervious surface cover "
                f"exacerbate the local urban heat island (UHI) effect."
            )

        if "intervention" in q or "mitigation" in q or "help" in q or "solution" in q:
            return (
                f"To mitigate heat in {zone}, we recommend: 1) Deploying Cool Roofs (est. cooling 1.0-2.0°C) due to the high built-up index ({built_up:.1f}%), "
                f"2) Creating shaded pocket parks (est. cooling 0.8-1.5°C) to protect the dense population, and "
                f"3) Planting green corridors to improve the low NDVI of {mean_ndvi:.3f}."
            )

        if "tree" in q or "plant" in q:
            suggested_trees = int(max(5000, min(50000, (0.3 - mean_ndvi) * 100000))) if mean_ndvi < 0.3 else 5000
            return (
                f"Based on a low vegetation coverage of {mean_ndvi:.3f} and population of {int(pop):,}, "
                f"we recommend planting approximately {suggested_trees:,} trees in {zone}. This intervention "
                f"is estimated to raise the local NDVI by 0.05-0.1, reducing surface temperatures by up to 1.5°C."
            )

        if "ndvi" in q or "vegetation" in q:
            return (
                f"An increase in NDVI by 0.1 across {zone} is projected to lower mean Land Surface Temperature (LST) "
                f"by approximately 1.2°C to 1.8°C. This increase will expand the urban canopy, enhance evaporative cooling, "
                f"and benefit {int(pop):,} residents."
            )

        if "built-up" in q or "builtup" in q or "concrete" in q:
            return (
                f"A 10% reduction in effective built-up area (through cool roofs, green roofs, or permeable pavement) in {zone} "
                f"is modeled to reduce LST by 0.8°C to 1.2°C. Shifting solar reflectance on these concrete surfaces "
                f"directly counteracts Delhi's extreme heat risk levels."
            )
            
        if "budget" in q or "allocate" in q or "cost" in q:
            lakh_pop = pop / 100000.0
            return (
                f"Our budget planner allocates funding to target high-severity zones. For {zone}, implementing the "
                f"suggested cool roofs and urban forestry interventions will cool approximately {lakh_pop:.1f} lakh residents "
                f"while maximizing thermal reduction efficiency."
            )

        return (
            f"The predicted temperature for {zone} is {predicted_lst:.2f}°C. Given its built-up percent of {built_up:.1f}% "
            f"and NDVI of {mean_ndvi:.3f}, policymakers should prioritize cool roofing and urban afforestation to cool "
            f"the {int(pop):,} residents."
        )
