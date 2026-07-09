from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    zone: str = Field(..., min_length=1)
    year: int = Field(..., ge=2018)
    month: int = Field(..., ge=1, le=12)


class PredictionResponse(BaseModel):
    zone: str
    year: int
    month: int
    predicted_lst: float
    historical_lst: float
    population: float
    population_density: float
    built_up_percent: float
    mean_ndvi: float
    risk_level: str
    recommendation: list[str]


class GeoSelectionRequest(BaseModel):
    longitude: float = Field(..., ge=-180, le=180)
    latitude: float = Field(..., ge=-90, le=90)


class GeoSelectionResponse(BaseModel):
    zone: str | None
    ward: str | None
    geometry: dict | None


class ChatRequest(BaseModel):
    zone: str = Field(..., min_length=1)
    year: int = Field(..., ge=2018)
    month: int = Field(..., ge=1, le=12)
    query: str = Field(..., min_length=1)


class BudgetRequest(BaseModel):
    budget: str = Field(..., min_length=1)
    year: int = Field(..., ge=2018)
    month: int = Field(..., ge=1, le=12)


class ReportRequest(BaseModel):
    zone: str = Field(..., min_length=1)
    year: int = Field(..., ge=2018)
    month: int = Field(..., ge=1, le=12)
    budget: str | None = None


class AssistantRequest(BaseModel):
    zone: str = Field(..., min_length=1)
    prediction: dict = Field(...)
    historical_data: list[dict] = Field(...)
    zone_statistics: dict = Field(...)
    question: str = Field(..., min_length=1)

