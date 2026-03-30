from fastapi import APIRouter

from serving.api.dependencies import Redis, Registry

router = APIRouter(tags=["health"])


@router.get("/health")
async def liveness():
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness(registry: Registry, redis: Redis):
    checks: dict[str, str] = {}

    try:
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    models = registry.list_models()
    checks["registry"] = f"{len(models)} models registered"

    all_ok = all("ok" in v or "registered" in v for v in checks.values())
    return {"status": "ready" if all_ok else "degraded", "checks": checks}


@router.get("/health/models")
async def list_models(registry: Registry):
    return {"models": registry.list_models()}
