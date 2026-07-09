from __future__ import annotations

from pathlib import Path
from fpdf import FPDF

from backend.services.csv_service import CSVService
from backend.services.prediction_service import PredictionService
from backend.services.analysis_service import AIAnalysisService


class ReportService:
    def __init__(self) -> None:
        self.csv_service = CSVService()
        self.prediction_service = PredictionService()
        self.analysis_service = AIAnalysisService()

    def generate_report(self, zone: str, year: int, month: int, budget: str | None = None) -> Path:
        summary = self.csv_service.get_zone_summary(zone)
        prediction = self.prediction_service.predict_for_zone(zone, year, month)
        analysis = self.analysis_service.explain_prediction(zone, year, month)

        report_text = [
            f"Zone: {zone}",
            f"Date: {month}/{year}",
            f"Predicted LST: {prediction.get('predicted_lst')}°C",
            f"Confidence: {analysis.get('confidence')}%",
            f"Verification Status: {prediction.get('verification_status', 'N/A')}",
            f"Verification Details: {prediction.get('verification_details', 'N/A')}",
            "\nZone Summary:",
            f"Population: {int(summary.get('population', 0)):,}",
            f"Population density: {int(summary.get('population_density', 0)):,} people/km²",
            f"Built-up %: {summary.get('built_up_percent'):.2f}%",
            f"NDVI: {summary.get('mean_ndvi'):.3f}",
            "\nRecommendations:",
        ]

        # Handle both list of dicts (structured) and list of strings (legacy/fallback)
        recs = prediction.get("recommendation", [])
        for rec in recs:
            if isinstance(rec, dict):
                title = rec.get("title", "Mitigation Strategy")
                desc = rec.get("description", "")
                impact = rec.get("expected_cooling_impact", "")
                cost = rec.get("estimated_implementation_cost", "")
                priority = rec.get("priority", "Medium")
                pop = rec.get("affected_population", 0)
                report_text.append(
                    f" - {title}: {desc}\n   Impact: {impact} | Cost: {cost} | Priority: {priority} | Population: {pop:,}"
                )
            else:
                report_text.append(f" - {rec}")

        report_text.extend([
            "\nHistorical Comparison:",
            analysis.get('historical_comparison', 'N/A'),
            "\nNeighbour Comparison:",
            analysis.get('neighbour_comparison', 'N/A'),
            "\nAnomalies Detected:",
        ])
        
        for anomaly in analysis.get('potential_anomalies', []):
            report_text.append(f" - {anomaly}")

        output_path = Path('backend/data/ai_report.pdf')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, f'UrbanCool AI Report - {zone}', ln=True)
        pdf.set_font('Helvetica', '', 11)
        pdf.multi_cell(0, 8, f'Date: {month}/{year}')
        pdf.ln(4)

        for line in report_text:
            pdf.multi_cell(0, 7, str(line))

        pdf.output(str(output_path))
        return output_path
