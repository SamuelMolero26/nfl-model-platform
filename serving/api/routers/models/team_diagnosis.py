from fastapi import APIRouter

from serving.api.dependencies import Redis, Registry, Settings
from serving.api.routers.models._common import PredictRequest, get_schema, run_prediction

router = APIRouter(tags=["models"])


@router.post("/team-diagnosis/predict")
async def predict(request: PredictRequest, registry: Registry, redis: Redis, settings: Settings):
    return await run_prediction("team_diagnosis", request, registry, redis, settings["cache"]["prediction_ttl"])


@router.get("/team-diagnosis/schema")
async def schema(registry: Registry):
    return await get_schema("team_diagnosis", registry)
