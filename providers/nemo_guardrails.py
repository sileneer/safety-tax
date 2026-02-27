"""
Condition B — NVIDIA NeMo Guardrails (Semantic / Dialog Control).

Uses Colang rules and embedding-based checks to gate inputs and outputs.
NeMo Guardrails sits as a wrapper around the LLM call and can block
requests before or after they reach the model.
"""

from __future__ import annotations

import asyncio
import logging

from nemoguardrails import RailsConfig, LLMRails

import config
from .base import BaseProvider, ProviderResult

logger = logging.getLogger(__name__)

# Canonical refusal messages defined in our Colang rules
_COLANG_REFUSAL_STRINGS = [
    "I'm sorry, I can't help with that request as it violates safety guidelines.",
    "I can't assist with that request.",
]

# Retry settings for rate-limit (429) errors.  NeMo makes ~3 internal LLM
# calls per test case, which can trigger Anthropic's 5 req/min limit.
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 30  # seconds — first retry waits 30s, then 60s, etc.


def _get_log_data(response, key: str, default=None):
    """Extract log data from a NeMo response, handling both dict and object forms."""
    # NeMo may return log as a dict key or as an attribute
    log = None
    if isinstance(response, dict):
        log = response.get("log")
    elif hasattr(response, "log"):
        log = response.log

    if log is None:
        return default

    if isinstance(log, dict):
        return log.get(key, default)
    else:
        return getattr(log, key, default)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a rate-limit (429) error from any layer."""
    msg = str(exc).lower()
    return "429" in msg or "rate_limit" in msg or "rate limit" in msg


class NeMoGuardrailsProvider(BaseProvider):
    name = "nemo_guardrails"

    def __init__(self):
        import os
        self.model = config.TARGET_MODEL

        # Ensure API keys are in the environment so NeMo/LangChain clients
        # can pick them up automatically (NeMo doesn't resolve ${VAR} in YAML).
        os.environ.setdefault("ANTHROPIC_API_KEY", config.ANTHROPIC_API_KEY)
        os.environ.setdefault("OPENAI_API_KEY", config.OPENAI_API_KEY)

        # Load Colang config from nemo_config/
        rails_config = RailsConfig.from_path(str(config.NEMO_CONFIG_DIR))
        self.rails = LLMRails(config=rails_config)

        # Enable detailed logging so we can extract token counts and rail activations.
        # NeMo accepts options as a plain dict; GenerationOptions may not forward
        # the `log` key, so we use a dict to match the documented API.
        self.generation_options = {
            "output_vars": ["triggered_input_rail"],
            "log": {
                "activated_rails": True,
                "llm_calls": True,
                "internal_events": True,
                "stats": True,
            },
        }

    async def process(self, prompt: str) -> ProviderResult:
        start = self._timer()
        input_tokens = 0
        output_tokens = 0

        try:
            # NeMo Guardrails orchestrates the full call (input check -> LLM -> output check)
            kwargs = {"messages": [{"role": "user", "content": prompt}]}
            if self.generation_options is not None:
                kwargs["options"] = self.generation_options

            # Retry with exponential backoff on rate-limit errors.
            # NeMo internally fires ~3 LLM calls per request, which can
            # burst past the Anthropic rate limit even with our outer throttle.
            response = None
            last_exc = None
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    response = await self.rails.generate_async(**kwargs)
                    break
                except Exception as exc:
                    if _is_rate_limit_error(exc) and attempt < _MAX_RETRIES:
                        wait = _RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            "Rate-limited on attempt %d/%d, retrying in %ds: %s",
                            attempt + 1, _MAX_RETRIES + 1, wait, exc,
                        )
                        await asyncio.sleep(wait)
                        last_exc = exc
                    else:
                        raise

            if response is None:
                raise last_exc  # type: ignore[misc]

            # Extract content — response may be a plain string, a dict,
            # or a GenerationResponse object (when GenerationOptions are used).
            if isinstance(response, str):
                raw_output = response
            elif isinstance(response, dict):
                raw_output = response.get("content", "")
            elif hasattr(response, "response"):
                # GenerationResponse: .response is a list of message dicts
                resp_msgs = response.response
                if isinstance(resp_msgs, list) and resp_msgs:
                    last_msg = resp_msgs[-1]
                    raw_output = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
                elif isinstance(resp_msgs, str):
                    raw_output = resp_msgs
                else:
                    raw_output = ""
                    logger.warning("Unexpected NeMo .response type: %s", type(resp_msgs))
            elif hasattr(response, "content"):
                raw_output = response.content or ""
            else:
                raw_output = str(response)
                logger.warning("NeMo response fell through to str(): type=%s", type(response))

            # Detect blocking via NeMo's structured data:
            # 1. Check if an input rail was triggered (returned in output_vars)
            triggered_input_rail = None
            if isinstance(response, dict):
                output_data = response.get("output_data") or {}
                triggered_input_rail = output_data.get("triggered_input_rail")
            elif hasattr(response, "output_data"):
                output_data = response.output_data or {}
                if isinstance(output_data, dict):
                    triggered_input_rail = output_data.get("triggered_input_rail")

            # 2. Check if the response exactly matches one of our Colang refusal strings
            is_colang_refusal = any(
                raw_output.strip() == refusal for refusal in _COLANG_REFUSAL_STRINGS
            )

            # 3. Check activated rails from the log
            activated_rails = _get_log_data(response, "activated_rails", []) or []

            blocked = bool(triggered_input_rail) or is_colang_refusal

            # Extract token counts from the NeMo response log.
            # Primary: use log.stats aggregate totals (most reliable).
            # Fallback: iterate log.llm_calls entries.
            log = None
            if hasattr(response, "log"):
                log = response.log
            elif isinstance(response, dict):
                log = response.get("log")

            if log is not None:
                # Try stats aggregate first
                stats = getattr(log, "stats", None) if not isinstance(log, dict) else (log.get("stats") if isinstance(log, dict) else None)
                if stats is not None:
                    if isinstance(stats, dict):
                        input_tokens = stats.get("llm_calls_total_prompt_tokens", 0) or 0
                        output_tokens = stats.get("llm_calls_total_completion_tokens", 0) or 0
                    else:
                        input_tokens = getattr(stats, "llm_calls_total_prompt_tokens", 0) or 0
                        output_tokens = getattr(stats, "llm_calls_total_completion_tokens", 0) or 0
                else:
                    # Fallback: iterate llm_calls entries (objects or dicts)
                    llm_calls_list = getattr(log, "llm_calls", None) if not isinstance(log, dict) else log.get("llm_calls")
                    for call in (llm_calls_list or []):
                        if isinstance(call, dict):
                            input_tokens += call.get("prompt_tokens", 0) or 0
                            output_tokens += call.get("completion_tokens", 0) or 0
                        else:
                            input_tokens += getattr(call, "prompt_tokens", 0) or 0
                            output_tokens += getattr(call, "completion_tokens", 0) or 0

            return ProviderResult(
                raw_output=raw_output,
                final_output=raw_output if not blocked else f"[BLOCKED] {raw_output}",
                blocked=blocked,
                latency_ms=self._timer() - start,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                metadata={
                    "nemo_response_type": type(response).__name__,
                    "triggered_input_rail": triggered_input_rail,
                    "activated_rails": activated_rails,
                    "token_counts_from_log": input_tokens + output_tokens > 0,
                },
            )

        except Exception as e:
            return ProviderResult(
                error=str(e),
                latency_ms=self._timer() - start,
            )
