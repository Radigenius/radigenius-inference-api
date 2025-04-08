from fastapi import Request
from fastapi.responses import JSONResponse
from app.exceptions.base import BaseException


async def custom_exception_handler(request: Request, exc: BaseException):
    return JSONResponse(
        status_code=exc.code,
        content={"message": exc.message, "errors": exc.errors, "key": exc.key},
    )