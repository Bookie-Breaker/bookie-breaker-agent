"""with_retries backoff, give-up, and non-retriable passthrough."""

import pytest

from agent.api.errors import DependencyError, NotFoundError
from agent.clients.retry import with_retries


class TestWithRetries:
    async def test_succeeds_after_transient_failures(self) -> None:
        calls = 0

        async def flaky() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise DependencyError("upstream 502")
            return "ok"

        result = await with_retries(flaky, attempts=3, base_delay=0.001)
        assert result == "ok"
        assert calls == 3

    async def test_gives_up_after_attempts(self) -> None:
        calls = 0

        async def always_down() -> str:
            nonlocal calls
            calls += 1
            raise DependencyError("upstream 502")

        with pytest.raises(DependencyError):
            await with_retries(always_down, attempts=3, base_delay=0.001)
        assert calls == 3

    async def test_not_found_is_never_retried(self) -> None:
        calls = 0

        async def missing() -> str:
            nonlocal calls
            calls += 1
            raise NotFoundError("no such game")

        with pytest.raises(NotFoundError):
            await with_retries(missing, attempts=3, base_delay=0.001)
        assert calls == 1

    async def test_first_success_needs_no_retry(self) -> None:
        async def fine() -> int:
            return 42

        assert await with_retries(fine, attempts=3) == 42
