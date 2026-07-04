"""Standard JSON envelope per api-contracts README: {data, meta{timestamp, request_id}}."""

import uuid
from contextvars import ContextVar
from datetime import UTC, datetime

from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class Meta(BaseModel):
    timestamp: datetime
    request_id: str


class Envelope[T](BaseModel):
    data: T
    meta: Meta


class Pagination(BaseModel):
    limit: int
    has_more: bool
    next_cursor: str | None = None


class PageMeta(Meta):
    pagination: Pagination


class PageEnvelope[T](BaseModel):
    data: list[T]
    meta: PageMeta


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, object] = {}


class ErrorEnvelope(BaseModel):
    error: ErrorBody
    meta: Meta


def make_meta() -> Meta:
    return Meta(timestamp=datetime.now(tz=UTC), request_id=request_id_var.get() or str(uuid.uuid4()))


def envelope[T](data: T) -> Envelope[T]:
    return Envelope[T](data=data, meta=make_meta())


def page_envelope[T](data: list[T], limit: int, has_more: bool, next_cursor: str | None = None) -> PageEnvelope[T]:
    meta = make_meta()
    page_meta = PageMeta(
        timestamp=meta.timestamp,
        request_id=meta.request_id,
        pagination=Pagination(limit=limit, has_more=has_more, next_cursor=next_cursor),
    )
    return PageEnvelope[T](data=data, meta=page_meta)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Accepts an inbound X-Request-ID or generates one, and echoes it back."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response
