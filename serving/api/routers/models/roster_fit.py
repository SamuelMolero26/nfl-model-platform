from fastapi import APIRouter

from serving.api.dependencies import Redis, Registry, Settings
from serving.api.routers.models._common import PredictRequest, get_schema, run_prediction

router = APIRouter(tags=["models"])


@router.post("/roster-fit/predict")
async def predict(request: PredictRequest, registry: Registry, redis: Redis, settings: Settings):
    return await run_prediction("roster_fit", request, registry, redis, settings["cache"]["prediction_ttl"])


@router.get("/roster-fit/schema")
async def schema(registry: Registry):
    return await get_schema("roster_fit", registry)
