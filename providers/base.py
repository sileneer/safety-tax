"""
Base provider interface.

Every configuration (Control, Guardrails AI, NeMo) must subclass BaseProvider
and implement `process()`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ProviderResult:
    """Standard result envelope returned by every provider."""
    raw_output: str = ""              # The model's raw response
    final_output: str = ""            # After guardrail post-processing (may == raw_output)
    blocked: bool = False             # True if the provider's native detection flagged a block
    standardized_blocked: bool | None = None  # Set by the judge based on response content analysis
    latency_ms: float = 0.0          # Wall-clock time for the full call
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0             # input + output + any internal validation tokens
    retries: int = 0                  # Number of re-prompts (relevant for Guardrails AI)
    error: str | None = None          # Non-None if the call errored out
    metadata: dict = field(default_factory=dict)


class BaseProvider(ABC):
    """Abstract base for all three experimental configurations."""

    name: str = "base"

    @abstractmethod
    async def process(self, prompt: str) -> ProviderResult:
        """Send a prompt through this configuration and return a ProviderResult."""
        ...

    @staticmethod
    def _timer() -> float:
        """High-res monotonic timestamp in ms."""
        return time.perf_counter() * 1000
