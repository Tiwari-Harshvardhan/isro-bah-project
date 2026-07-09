from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from backend.routers.schemas import BudgetRequest, ReportRequest
from backend.services.budget_service import BudgetPlannerService
from backend.services.report_service import ReportService

router = APIRouter()
budget_service = BudgetPlannerService()
report_service = ReportService()


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
        output_path = report_service.generate_report(request.zone, request.year, request.month, request.budget)
        return FileResponse(path=output_path, filename=f"{request.zone.replace(' ', '_')}_report.pdf", media_type='application/pdf')
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Report generation failed: {exc}') from exc
