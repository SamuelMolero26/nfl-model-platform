import json
from typing import Any, Optional

import redis.asyncio as aioredis
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from serving.api.cache import keys as cache_keys
from serving.models.registry import ModelRegistry


class PredictRequest(BaseModel):
    inputs: dict[str, Any]
    version: Optional[str] = None


async def run_prediction(
    model_name: str,
    request: PredictRequest,
    registry: ModelRegistry,
    redis: aioredis.Redis,
    ttl: int,
) -> dict:
    cache_key = cache_keys.prediction_key(model_name, request.inputs)

    cached = await redis.get(cache_key)
    if cached:
        result = json.loads(cached)
        result["cached"] = True
        return result

    try:
        result = await run_in_threadpool(
            registry.predict, model_name, request.inputs, request.version
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    await redis.setex(cache_key, ttl, json.dumps(result, default=str))
    result["cached"] = False
    return result


async def get_schema(model_name: str, registry: ModelRegistry) -> dict:
    try:
        model = registry.get(model_name)
        return {
            "model": model_name,
            "features": model.feature_names,
            "metadata": model.metadata,
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
