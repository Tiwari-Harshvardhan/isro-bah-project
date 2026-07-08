from fastapi import APIRouter, HTTPException

from backend.api.schemas import ChatRequest, PredictionRequest
from backend.services.analysis_service import AIAnalysisService
from backend.services.chat_service import AIChatService

router = APIRouter()
analysis_service = AIAnalysisService()
chat_service = AIChatService()


@router.post("/analysis/explain")
def explain_prediction(request: PredictionRequest) -> dict[str, object]:
    try:
        return analysis_service.explain_prediction(request.zone, request.year, request.month)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Analysis failed") from exc


@router.post("/analysis/chat")
def chat_prediction(request: ChatRequest) -> dict[str, object]:
    try:
        return chat_service.answer_query(request.zone, request.year, request.month, request.query)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Chat analysis failed") from exc
