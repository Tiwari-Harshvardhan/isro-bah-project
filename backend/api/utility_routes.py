from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from fpdf import FPDF
from pathlib import Path

from backend.api.schemas import BudgetRequest, ReportRequest
from backend.services.budget_service import BudgetPlannerService
from backend.services.csv_service import CSVService
from backend.services.prediction_service import PredictionService
from backend.services.analysis_service import AIAnalysisService

router = APIRouter()
budget_service = BudgetPlannerService()
analysis_service = AIAnalysisService()
csv_service = CSVService()
prediction_service = PredictionService()


@router.post('/budget/plan')
def plan_budget(request: BudgetRequest) -> dict[str, object]:
    try:
        amount_text = request.budget.replace('₹', '').replace('Cr', '').replace('crore', '').strip()
        amount = float(amount_text) * 10000000 if 'crore' in request.budget.lower() or 'cr' in request.budget.lower() else float(amount_text)
        return budget_service.plan_budget(amount, request.year, request.month)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Budget planning failed: {exc}') from exc


@router.post('/report/generate')
def generate_report(request: ReportRequest) -> FileResponse:
    try:
        summary = csv_service.get_zone_summary(request.zone)
        prediction = prediction_service.predict_for_zone(request.zone, request.year, request.month)
        analysis = analysis_service.explain_prediction(request.zone, request.year, request.month)

        report_text = [
            f"Zone: {request.zone}",
            f"Date: {request.month}/{request.year}",
            f"Predicted LST: {prediction['predicted_lst']}°C",
            f"Confidence: {analysis['confidence']}%",
            "\nZone Summary:",
            f"Population: {summary['population']}",
            f"Population density: {summary['population_density']}",
            f"Built-up %: {summary['built_up_percent']}",
            f"NDVI: {summary['mean_ndvi']}",
            "\nRecommendations:",
            *analysis['reasoning'],
            "\nHistorical Comparison:",
            analysis['historical_comparison'],
            "\nNeighbour Comparison:",
            analysis['neighbour_comparison'],
            "\nAnomalies:",
            *analysis['potential_anomalies'],
        ]

        output_path = Path('backend/data/ai_report.pdf')
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, f'UrbanCool AI Report - {request.zone}', ln=True)
        pdf.set_font('Helvetica', '', 11)
        pdf.multi_cell(0, 8, f'Date: {request.month}/{request.year}')
        pdf.ln(4)

        for line in report_text:
            if isinstance(line, str):
                pdf.multi_cell(0, 7, str(line))
            else:
                pdf.multi_cell(0, 7, str(line))

        pdf.output(str(output_path))
        return FileResponse(path=output_path, filename=f"{request.zone.replace(' ', '_')}_report.pdf", media_type='application/pdf')
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Report generation failed: {exc}') from exc
