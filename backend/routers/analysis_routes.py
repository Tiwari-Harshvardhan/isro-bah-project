from fastapi import APIRouter, HTTPException

from backend.routers.schemas import ChatRequest, PredictionRequest, AssistantRequest
from backend.services.analysis_service import AIAnalysisService
from backend.services.chat_service import AIChatService
from backend.agents.planning_assistant import PlanningAssistantAgent
from backend.config import GEMINI_API_KEY

router = APIRouter()
analysis_service = AIAnalysisService()
chat_service = AIChatService()
assistant_agent = PlanningAssistantAgent(api_key=GEMINI_API_KEY)


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


@router.post("/assistant")
def assistant(request: AssistantRequest) -> dict[str, object]:
    try:
        answer = assistant_agent.answer_question(
            zone=request.zone,
            prediction=request.prediction,
            historical_data=request.historical_data,
            zone_statistics=request.zone_statistics,
            question=request.question
        )
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Assistant failed: {exc}") from exc
