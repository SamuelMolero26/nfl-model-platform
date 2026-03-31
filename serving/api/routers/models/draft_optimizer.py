import json
import uuid

from fastapi import APIRouter, BackgroundTasks
from fastapi.concurrency import run_in_threadpool

from serving.api.cache import keys as cache_keys
from serving.api.dependencies import Redis, Registry, Settings
from serving.api.routers.models._common import (
    PredictRequest,
    get_schema,
    run_prediction,
)

router = APIRouter(tags=["models"])


@router.post("/draft-optimizer/predict")
async def predict_sync(
    request: PredictRequest, registry: Registry, redis: Redis, settings: Settings
):
    """Synchronous draft optimizer prediction — blocks until the CVXPY solve completes."""
    return await run_prediction(
        "draft_optimizer", request, registry, redis, settings["cache"]["prediction_ttl"]
    )


@router.post("/draft-optimizer/jobs")
async def enqueue_job(
    request: PredictRequest,
    background_tasks: BackgroundTasks,
    registry: Registry,
    redis: Redis,
    settings: Settings,
):
    """Enqueue an async draft optimizer job. Returns a job_id to poll for results."""
    job_id = str(uuid.uuid4())
    key = cache_keys.job_key(job_id)
    ttl = settings["cache"]["job_ttl"]

    await redis.setex(key, ttl, json.dumps({"status": "queued"}))
    background_tasks.add_task(
        _run_job, job_id, request.inputs, request.version, registry, redis, ttl
    )
    return {"job_id": job_id, "status": "queued"}


@router.get("/draft-optimizer/jobs/{job_id}")
async def poll_job(job_id: str, redis: Redis):
    """Poll the result of an async draft optimizer job."""
    key = cache_keys.job_key(job_id)
    data = await redis.get(key)
    if data is None:
        return {"job_id": job_id, "status": "not_found"}
    payload = json.loads(data)
    return {"job_id": job_id, **payload}


@router.get("/draft-optimizer/schema")
async def schema(registry: Registry):
    return await get_schema("draft_optimizer", registry)


async def _run_job(job_id: str, inputs: dict, version, registry, redis, ttl: int):
    key = cache_keys.job_key(job_id)
    await redis.setex(key, ttl, json.dumps({"status": "running"}))
    try:
        result = await run_in_threadpool(
            registry.predict, "draft_optimizer", inputs, version
        )
        await redis.setex(
            key, ttl, json.dumps({"status": "complete", "result": result}, default=str)
        )
    except Exception as e:
        await redis.setex(key, ttl, json.dumps({"status": "error", "error": str(e)}))
