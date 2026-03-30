from fastapi import APIRouter

from serving.api.dependencies import Redis, Registry, Settings
from serving.api.routers.models._common import PredictRequest, get_schema, run_prediction

router = APIRouter(tags=["models"])


@router.post("/health-analyzer/predict")
async def predict(request: PredictRequest, registry: Registry, redis: Redis, settings: Settings):
    return await run_prediction("health_analyzer", request, registry, redis, settings["cache"]["prediction_ttl"])


@router.get("/health-analyzer/schema")
async def schema(registry: Registry):
    return await get_schema("health_analyzer", registry)
