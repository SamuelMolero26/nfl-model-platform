import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI

from serving.api.cache.client import create_redis_pool
from serving.api.errors import (
    generic_error_handler,
    key_error_handler,
)

from serving.api.routers import health
from serving.api.routers.models import (
    career_simulator,
    draft_optimizer,
    health_analyzer,
    player_projection,
    positional_flexibility,
    roster_fit,
    team_diagnosis,
)
from serving.models.registry import ModelRegistry


def _load_settings() -> dict:
    config_path = Path(os.environ.get("API_CONFIG_PATH", "config/api.yaml"))
    with open(config_path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = _load_settings()
    app.state.settings = settings

    redis_pool = create_redis_pool(settings["redis"]["pool_size"])
    await redis_pool.ping()
    app.state.redis = redis_pool

    app.state.registry = ModelRegistry.instance()

    yield

    await redis_pool.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="NFL Model Platform API",
        version="1.0.0",
        lifespan=lifespan,
        root_path=os.environ.get("ROOT_PATH", ""),
    )

    app.add_exception_handler(KeyError, key_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)

    app.include_router(health.router)
    app.include_router(player_projection.router)
    app.include_router(positional_flexibility.router)
    app.include_router(team_diagnosis.router)
    app.include_router(draft_optimizer.router)
    app.include_router(career_simulator.router)
    app.include_router(roster_fit.router)
    app.include_router(health_analyzer.router)

    return app


app = create_app()
