from fastapi import APIRouter, HTTPException

from backend.api.schemas import GeoSelectionRequest, PredictionRequest
from backend.services.geo_service import GeoService
from backend.services.prediction_service import PredictionService

router = APIRouter()
geo_service = GeoService()
prediction_service = PredictionService()


@router.get("/")
def root() -> dict[str, str]:
    return {"message": "UrbanCool AI backend is running"}


@router.get("/zones")
def get_zones() -> list[str]:
    return geo_service.get_zone_names()


@router.get("/zone/{zone_name}")
def get_zone_summary(zone_name: str) -> dict[str, object]:
    try:
        return prediction_service.csv_service.get_zone_summary(zone_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/predict", response_model=dict)
def predict(request: PredictionRequest) -> dict[str, object]:
    try:
        return prediction_service.predict_for_zone(request.zone, request.year, request.month)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@router.get("/zone/{zone_name}/history")
def get_zone_history(zone_name: str) -> list[dict[str, object]]:
    try:
        history = prediction_service.csv_service.get_zone_history(zone_name)
        return history.to_dict(orient="records")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/geo/select")
def select_geometry(payload: GeoSelectionRequest) -> dict[str, object]:
    try:
        return geo_service.select_geometry(payload.longitude, payload.latitude)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/geo/geojson")
def get_geojson() -> dict[str, object]:
    try:
        return geo_service.get_map_geojson()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="GeoJSON generation failed") from exc
