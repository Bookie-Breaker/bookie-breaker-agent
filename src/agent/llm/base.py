"""Provider-neutral LLM interface (ADR-011).

Two model tiers keep costs sane: "quality" for user-facing analyses,
"cheap" for routine summaries and alert descriptions. Ollama deployments
may map both tiers to the same local model.
"""

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


class LLMProvider(Protocol):
    provider_name: str

    def model_for(self, tier: ModelTier) -> str: ...

    async def complete(
        self, *, system: str, prompt: str, tier: ModelTier = "quality", max_tokens: int | None = None
    ) -> LLMResult: ...

    async def is_healthy(self) -> bool: ...

    async def aclose(self) -> None: ...
