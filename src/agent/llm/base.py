"""Provider-neutral LLM interface (ADR-011).

Two model tiers keep costs sane: "quality" for user-facing analyses,
"cheap" for routine summaries and alert descriptions. Ollama deployments
may map both tiers to the same local model.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

ModelTier = Literal["quality", "cheap"]


class LLMError(Exception):
    """Any provider failure; the API layer maps this to DependencyError."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    provider: str
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class LLMStreamChunk:
    """One streamed delta; the terminal chunk carries the usage-bearing result.

    Providers yield text deltas as they arrive and finish with a chunk whose
    final is the same LLMResult complete() would have returned (full text,
    model, token counts), so callers account tokens identically on both paths.
    """

    text: str
    final: LLMResult | None = None


class LLMProvider(Protocol):
    provider_name: str

    def model_for(self, tier: ModelTier) -> str: ...

    async def complete(
        self, *, system: str, prompt: str, tier: ModelTier = "quality", max_tokens: int | None = None
    ) -> LLMResult: ...

    def stream(
        self, *, system: str, prompt: str, tier: ModelTier = "quality", max_tokens: int | None = None
    ) -> AsyncIterator[LLMStreamChunk]: ...

    async def is_healthy(self) -> bool: ...

    async def aclose(self) -> None: ...
