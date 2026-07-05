"""Shared envelope-unwrapping HTTP client base.

GETs are idempotent and retried on transient failures (network, 5xx,
timeouts). POSTs are single-shot unless the caller opts in with
retriable=True — safe only when the upstream endpoint is idempotent
(e.g. simulation runs and prediction batches keyed by game).
"""

from typing import Any

import httpx

from agent.api.errors import (
    DependencyError,
    DependencyTimeoutError,
    NotFoundError,
    UnprocessableError,
)
from agent.clients.retry import with_retries


class ServiceClient:
    service_name = "upstream"

    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def get_data(self, path: str, resource: str, params: dict[str, Any] | None = None) -> Any:
        """GET an enveloped endpoint and return its data payload.

        Returns dict or list depending on the endpoint. Raises NotFoundError
        on 404, DependencyError/DependencyTimeoutError on upstream failures
        (after transparent retries).
        """
        return await with_retries(lambda: self._get_once(path, resource, params))

    async def _get_once(self, path: str, resource: str, params: dict[str, Any] | None) -> Any:
        url = f"{self._base_url}{path}"
        try:
            response = await self._client.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise DependencyTimeoutError(f"{self.service_name} timed out fetching {resource}") from exc
        except httpx.HTTPError as exc:
            raise DependencyError(f"{self.service_name} is unavailable: {exc}") from exc
        if response.status_code == 404:
            raise NotFoundError(f"{resource} not found in {self.service_name}")
        if response.status_code >= 500:
            raise DependencyError(f"{self.service_name} returned {response.status_code} for {resource}")
        payload: dict[str, Any] = response.json()
        if "data" not in payload:
            raise DependencyError(f"{self.service_name} returned a malformed envelope for {resource}")
        return payload["data"]

    async def post_data(
        self,
        path: str,
        resource: str,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
        retriable: bool = False,
    ) -> Any:
        """POST to an enveloped endpoint and return its data payload.

        Accepts 200/201/202. Raises NotFoundError on 404, UnprocessableError
        on 422 (with the upstream message when available), and
        DependencyError/DependencyTimeoutError on other upstream failures.
        retriable=True retries transient failures — only for idempotent
        upstream endpoints; bet placement must stay single-shot.
        """
        if retriable:
            return await with_retries(lambda: self._post_once(path, resource, json, headers))
        return await self._post_once(path, resource, json, headers)

    async def _post_once(
        self,
        path: str,
        resource: str,
        json: dict[str, Any],
        headers: dict[str, str] | None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        try:
            response = await self._client.post(url, json=json, headers=headers)
        except httpx.TimeoutException as exc:
            raise DependencyTimeoutError(f"{self.service_name} timed out creating {resource}") from exc
        except httpx.HTTPError as exc:
            raise DependencyError(f"{self.service_name} is unavailable: {exc}") from exc
        if response.status_code in (200, 201, 202):
            payload: dict[str, Any] = response.json()
            if "data" not in payload:
                raise DependencyError(f"{self.service_name} returned a malformed envelope for {resource}")
            return payload["data"]
        message = self._upstream_message(response)
        if response.status_code == 404:
            raise NotFoundError(f"{resource} not found in {self.service_name}: {message}")
        if response.status_code == 422:
            raise UnprocessableError(f"{self.service_name} rejected {resource}: {message}")
        raise DependencyError(f"{self.service_name} returned {response.status_code} for {resource}: {message}")

    @staticmethod
    def _upstream_message(response: httpx.Response) -> str:
        try:
            error = response.json().get("error", {})
            return str(error.get("message", response.text))
        except ValueError:
            return response.text

    async def is_healthy(self, health_path: str) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}{health_path}", timeout=1.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200
