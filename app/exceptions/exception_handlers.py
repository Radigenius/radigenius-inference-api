import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from app.exceptions.base import BaseException

logger = logging.getLogger(__name__)

async def custom_exception_handler(request: Request, exc: BaseException):
    logger.error(f"{exc.key} | {exc.message} | {exc.errors}")
    
    return JSONResponse(
        status_code=exc.code,
        content={"message": exc.message, "errors": exc.errors, "key": exc.key},
    )