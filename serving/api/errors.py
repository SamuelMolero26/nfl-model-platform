import uuid
from datetime import datetime, timezone

from fastapi import Request
from fastapi.responses import JSONResponse

from serving.data_lake_client import (
    DataLakeConnectionError,
    DataLakeNotFoundError,
    DataLakeQueryError,
)


def _body(error: str, message: str) -> dict:
    return {
        "error": error,
        "message": message,
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def not_found_handler(request: Request, exc: DataLakeNotFoundError):
    return JSONResponse(status_code=404, content=_body("not_found", str(exc)))


async def connection_error_handler(request: Request, exc: DataLakeConnectionError):
    return JSONResponse(status_code=502, content=_body("data_lake_unavailable", str(exc)))


async def query_error_handler(request: Request, exc: DataLakeQueryError):
    return JSONResponse(status_code=422, content=_body("query_error", str(exc)))


async def key_error_handler(request: Request, exc: KeyError):
    return JSONResponse(status_code=404, content=_body("not_found", str(exc)))


async def generic_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=_body("internal_error", "An unexpected error occurred."),
    )
