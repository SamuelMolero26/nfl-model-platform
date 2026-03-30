from typing import Annotated

import redis.asyncio as aioredis
from fastapi import Depends, Request

from serving.models.registry import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_redis(request: Request) -> aioredis.Redis:
    return request.app.state.redis


def get_settings(request: Request) -> dict:
    return request.app.state.settings


Registry = Annotated[ModelRegistry, Depends(get_registry)]
Redis = Annotated[aioredis.Redis, Depends(get_redis)]
Settings = Annotated[dict, Depends(get_settings)]
