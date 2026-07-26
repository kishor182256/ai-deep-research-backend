from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.core.config import settings

logger = logging.getLogger(__name__)


ERROR_MESSAGES: dict[int, str] = {
    status.HTTP_400_BAD_REQUEST: "We could not process that request. Please check the details and try again.",
    status.HTTP_401_UNAUTHORIZED: "Please sign in before continuing.",
    status.HTTP_403_FORBIDDEN: "You do not have permission to perform this action.",
    status.HTTP_404_NOT_FOUND: "We could not find the requested resource.",
    status.HTTP_405_METHOD_NOT_ALLOWED: "This action is not available for the requested endpoint.",
    status.HTTP_409_CONFLICT: "This action conflicts with existing data.",
    status.HTTP_422_UNPROCESSABLE_CONTENT: "Some required information is missing or invalid.",
    status.HTTP_429_TOO_MANY_REQUESTS: "Too many requests. Please wait a moment and try again.",
    status.HTTP_500_INTERNAL_SERVER_ERROR: "Something went wrong on our side. Please try again shortly.",
    status.HTTP_503_SERVICE_UNAVAILABLE: "The service is temporarily unavailable. Please try again shortly.",
}


class AppException(Exception):
    def __init__(
        self,
        *,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        code: str = "application_error",
        details: Any | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details
        super().__init__(message)


def register_exception_handlers(app: FastAPI) -> None:
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        status_code = exc.status_code
        message = _friendly_http_message(status_code=status_code, detail=exc.detail)
        return _error_response(
            request=request,
            status_code=status_code,
            code=_code_for_status(status_code),
            message=message,
            details=_safe_http_details(exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="validation_error",
            message=ERROR_MESSAGES[status.HTTP_422_UNPROCESSABLE_CONTENT],
            details=_format_validation_errors(exc.errors()),
        )

    @app.exception_handler(ResponseValidationError)
    async def response_validation_exception_handler(request: Request, exc: ResponseValidationError) -> JSONResponse:
        logger.exception(
            "Response validation failed",
            extra={"request_id": _request_id(request), "path": str(request.url.path)},
        )
        return _error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="response_validation_error",
            message=ERROR_MESSAGES[status.HTTP_500_INTERNAL_SERVER_ERROR],
            details=_debug_details(exc.errors()),
        )

    @app.exception_handler(IntegrityError)
    async def integrity_exception_handler(request: Request, exc: IntegrityError) -> JSONResponse:
        logger.warning(
            "Database integrity error",
            exc_info=exc,
            extra={"request_id": _request_id(request), "path": str(request.url.path)},
        )
        return _error_response(
            request=request,
            status_code=status.HTTP_409_CONFLICT,
            code="database_conflict",
            message=ERROR_MESSAGES[status.HTTP_409_CONFLICT],
            details=_debug_details(str(exc.orig)),
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.exception(
            "Database error",
            extra={"request_id": _request_id(request), "path": str(request.url.path)},
        )
        return _error_response(
            request=request,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="database_unavailable",
            message="We are having trouble reaching the database. Please try again shortly.",
            details=_debug_details(str(exc)),
        )

    @app.exception_handler(OSError)
    async def os_error_handler(request: Request, exc: OSError) -> JSONResponse:
        if _looks_like_database_connection_error(exc):
            logger.exception(
                "Database connection error",
                extra={"request_id": _request_id(request), "path": str(request.url.path)},
            )
            return _error_response(
                request=request,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code="database_unavailable",
                message="We are having trouble reaching the database. Please start PostgreSQL and try again.",
                details=_debug_details(str(exc)),
            )

        logger.exception(
            "Operating system error",
            extra={"request_id": _request_id(request), "path": str(request.url.path)},
        )
        return _error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="system_error",
            message=ERROR_MESSAGES[status.HTTP_500_INTERNAL_SERVER_ERROR],
            details=_debug_details(str(exc)),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled application error",
            extra={"request_id": _request_id(request), "path": str(request.url.path)},
        )
        return _error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="internal_server_error",
            message=ERROR_MESSAGES[status.HTTP_500_INTERNAL_SERVER_ERROR],
            details=_debug_details(str(exc)),
        )


def _error_response(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Any | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    payload: dict[str, Any] = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "details": details,
            "path": request.url.path,
            "request_id": request_id,
        },
    }
    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers={"X-Request-ID": request_id},
    )


def _friendly_http_message(*, status_code: int, detail: Any) -> str:
    framework_defaults = {"Not Found", "Method Not Allowed"}
    if (
        isinstance(detail, str)
        and detail
        and detail not in framework_defaults
        and status_code < status.HTTP_500_INTERNAL_SERVER_ERROR
    ):
        return detail
    return ERROR_MESSAGES.get(status_code, ERROR_MESSAGES[status.HTTP_500_INTERNAL_SERVER_ERROR])


def _safe_http_details(detail: Any) -> Any | None:
    if isinstance(detail, (dict, list)):
        return detail
    return None


def _format_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, str]]:
    formatted_errors: list[dict[str, str]] = []
    for error in errors:
        location = ".".join(str(part) for part in error.get("loc", []) if part != "body")
        formatted_errors.append(
            {
                "field": location or "request",
                "message": str(error.get("msg", "Invalid value")),
                "type": str(error.get("type", "validation_error")),
            }
        )
    return formatted_errors


def _debug_details(details: Any) -> Any | None:
    return details if settings.app_debug else None


def _code_for_status(status_code: int) -> str:
    return {
        status.HTTP_400_BAD_REQUEST: "bad_request",
        status.HTTP_401_UNAUTHORIZED: "unauthorized",
        status.HTTP_403_FORBIDDEN: "forbidden",
        status.HTTP_404_NOT_FOUND: "not_found",
        status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
        status.HTTP_409_CONFLICT: "conflict",
        status.HTTP_422_UNPROCESSABLE_CONTENT: "validation_error",
        status.HTTP_429_TOO_MANY_REQUESTS: "rate_limited",
        status.HTTP_503_SERVICE_UNAVAILABLE: "service_unavailable",
    }.get(status_code, "http_error")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid4()))


def _looks_like_database_connection_error(exc: OSError) -> bool:
    text = str(exc).lower()
    return (
        "connect call failed" in text
        or "connection refused" in text
        or "errno 10061" in text
        or "127.0.0.1', 5432" in text
        or "::1', 5432" in text
    )
