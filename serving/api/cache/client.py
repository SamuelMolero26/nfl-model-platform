import os

import redis.asyncio as aioredis


def create_redis_pool(pool_size: int = 10) -> aioredis.Redis:
    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    return aioredis.from_url(
        url,
        max_connections=pool_size,
        decode_responses=True,
    )
