import uuid
from datetime import datetime, timezone

from fastapi import Request
from fastapi.responses import JSONResponse


def _body(error: str, message: str) -> dict:
    return {
        "error": error,
        "message": message,
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def key_error_handler(request: Request, exc: KeyError):
    return JSONResponse(status_code=404, content=_body("not_found", str(exc)))


async def connection_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=503,
        content=_body("service_unavailable", str(exc)),
    )


async def generic_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=_body("internal_error", "An unexpected error occurred."),
    )
