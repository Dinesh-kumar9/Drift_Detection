"""
Rate limiting for every API endpoint (fixed one-minute windows).

Each request is counted against exactly one bucket, chosen by the first rule
that matches its method and path:

    auth     POST /auth/login, /auth/register        per client IP   (password guessing)
    predict  POST /models/{id}/predict               per user        (serving traffic)
    jobs     upload, train, drift check, simulate,   per user        (work that costs compute)
             retrain trigger / approve
    api      everything else                         per user

Requests without a valid token are counted per client IP. Counters live in
Redis so every API process shares them. If Redis is unreachable the limiter
lets requests through (fails open): losing the counter store must not take
serving down with it.
"""

import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import redis.asyncio as aioredis
from jose import JWTError, jwt
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import settings

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60
EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}


# ─── Rules ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Rule:
    scope: str
    label: str  # used in the 429 message
    limit: Callable[[], int]  # read at request time so settings changes apply immediately
    per_ip: bool = False
    method: str = "POST"
    path: Optional[re.Pattern] = None  # None matches every path

    def matches(self, method: str, path: str) -> bool:
        if self.path is None:
            return True
        return method == self.method and bool(self.path.match(path))


_JOB_PATHS = re.compile(
    r"^/(datasets/?|experiments/train/?|observability/[^/]+/(check|simulate)/?"
    r"|retrain/[^/]+/trigger/?|retrain/jobs/[^/]+/approve/?)$"
)

RULES = (
    Rule(
        "auth",
        "sign-in attempts",
        lambda: settings.RATE_LIMIT_AUTH_PER_MINUTE,
        per_ip=True,
        path=re.compile(r"^/auth/(login|register)/?$"),
    ),
    Rule(
        "predict",
        "predictions",
        lambda: settings.RATE_LIMIT_PREDICT_PER_MINUTE,
        path=re.compile(r"^/models/[^/]+/predict/?$"),
    ),
    Rule(
        "jobs",
        "jobs (uploads, training, drift checks, retraining)",
        lambda: settings.RATE_LIMIT_JOBS_PER_MINUTE,
        path=_JOB_PATHS,
    ),
    Rule("api", "API requests", lambda: settings.RATE_LIMIT_DEFAULT_PER_MINUTE),
)


def match_rule(method: str, path: str) -> Rule:
    return next(rule for rule in RULES if rule.matches(method, path))


# ─── Counter stores ───────────────────────────────────────────────────────────


class Store(Protocol):
    async def hit(self, key: str) -> int:
        """Count one request against `key` and return the total in the current window."""


class MemoryStore:
    """Single-process store for tests and local runs without Redis."""

    def __init__(self) -> None:
        self._window: Optional[int] = None
        self._counts: dict[str, int] = {}

    async def hit(self, key: str) -> int:
        window = int(time.time() // WINDOW_SECONDS)
        if window != self._window:  # keys embed the window, so old counts can be dropped wholesale
            self._window, self._counts = window, {}
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def reset(self) -> None:
        self._window, self._counts = None, {}


class RedisStore:
    def __init__(self, url: str) -> None:
        self._client = aioredis.from_url(url, socket_connect_timeout=0.25, socket_timeout=0.25)

    async def hit(self, key: str) -> int:
        pipe = self._client.pipeline(transaction=True)
        pipe.incr(key)
        pipe.expire(key, WINDOW_SECONDS + 5)
        count, _ = await pipe.execute()
        return int(count)


_stores: dict[tuple[str, str], Store] = {}
_last_store_warning = 0.0


def get_store() -> Store:
    kind = settings.RATE_LIMIT_STORAGE
    key = (kind, settings.REDIS_URL)
    if key not in _stores:
        _stores[key] = MemoryStore() if kind == "memory" else RedisStore(settings.REDIS_URL)
    return _stores[key]


def reset_memory_store() -> None:
    for store in _stores.values():
        if isinstance(store, MemoryStore):
            store.reset()


# ─── Identity ─────────────────────────────────────────────────────────────────


def client_ip(request: Request) -> str:
    if settings.RATE_LIMIT_TRUST_FORWARDED_FOR:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def identity(request: Request, rule: Rule) -> str:
    """Users are counted by account; anonymous or invalid-token requests by IP."""
    if not rule.per_ip:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            try:
                payload = jwt.decode(auth[7:], settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
                if payload.get("sub"):
                    return f"user:{payload['sub']}"
            except JWTError:
                pass  # the endpoint itself rejects the token; count it against the IP
    return f"ip:{client_ip(request)}"


# ─── Middleware ───────────────────────────────────────────────────────────────


class RateLimitMiddleware:
    """Pure ASGI middleware: BaseHTTPMiddleware would add a task hop to every request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not settings.RATE_LIMIT_ENABLED:
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        if request.method == "OPTIONS" or request.url.path in EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        rule = match_rule(request.method, request.url.path)
        limit = rule.limit()
        now = time.time()
        key = f"ratelimit:{rule.scope}:{identity(request, rule)}:{int(now // WINDOW_SECONDS)}"
        count = await _hit(key)
        if count is None:  # store unavailable → fail open
            await self.app(scope, receive, send)
            return

        reset_in = max(1, math.ceil(WINDOW_SECONDS - now % WINDOW_SECONDS))
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(max(0, limit - count)),
            "X-RateLimit-Reset": str(reset_in),
        }
        if count > limit:
            detail = f"Too many {rule.label}: the limit is {limit} per minute. Try again in {reset_in}s."
            response = JSONResponse(
                status_code=429, content={"detail": detail}, headers={**headers, "Retry-After": str(reset_in)}
            )
            await response(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).update(headers)
            await send(message)

        await self.app(scope, receive, send_with_headers)


async def _hit(key: str) -> Optional[int]:
    global _last_store_warning
    try:
        return await get_store().hit(key)
    except Exception as exc:  # noqa: BLE001 — any store failure must fail open
        if time.monotonic() - _last_store_warning > WINDOW_SECONDS:
            _last_store_warning = time.monotonic()
            logger.warning("Rate limiting disabled: counter store unavailable (%s)", exc)
        return None
