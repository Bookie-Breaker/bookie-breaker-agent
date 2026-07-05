"""Small retry helper for transient upstream failures.

Retries DependencyError/DependencyTimeoutError (network, 5xx, timeouts)
with exponential backoff and jitter. NotFoundError and UnprocessableError
are semantic responses, not transients, and are never retried.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

from agent.api.errors import DependencyError, DependencyTimeoutError

logger = logging.getLogger(__name__)

RETRIABLE_ERRORS: tuple[type[Exception], ...] = (DependencyError, DependencyTimeoutError)


async def with_retries[T](
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
    retry_on: tuple[type[Exception], ...] = RETRIABLE_ERRORS,
) -> T:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except retry_on as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            delay = min(base_delay * (2**attempt), max_delay) * (0.5 + random.random() / 2)  # noqa: S311
            logger.info("retrying after %s (attempt %d/%d, %.2fs)", type(exc).__name__, attempt + 1, attempts, delay)
            await asyncio.sleep(delay)
    assert last_error is not None  # loop always sets it before breaking
    raise last_error
