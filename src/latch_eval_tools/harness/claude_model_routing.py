from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ClaudeModelUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    web_search_requests: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    context_window: int | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, ge=0)
    canonical_model: str | None = None
    provider: str | None = None


class ClaudeSafetyFallback(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_subtype: Literal["model_refusal_fallback"] = "model_refusal_fallback"
    original_model: str
    fallback_model: str
    trigger: str | None = None
    category: str | None = None
    explanation: str | None = None
    session_id: str | None = None


class ClaudeModelRouting(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    requested_model: str
    switch_models_on_flag: bool
    initial_model: str | None = None
    effective_model: str | None = None
    fallback_occurred: bool
    fallback_recovered: bool
    terminal_stop_reason: str | None = None
    safety_fallbacks: list[ClaudeSafetyFallback] = Field(default_factory=list)
    per_model_usage: dict[str, ClaudeModelUsage] = Field(default_factory=dict)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _nonnegative_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        return 0.0
    return parsed


def _maximum_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _usage_from_record(record: dict[str, Any]) -> ClaudeModelUsage:
    return ClaudeModelUsage(
        input_tokens=_nonnegative_int(record.get("inputTokens")),
        output_tokens=_nonnegative_int(record.get("outputTokens")),
        cache_read_input_tokens=_nonnegative_int(record.get("cacheReadInputTokens")),
        cache_creation_input_tokens=_nonnegative_int(
            record.get("cacheCreationInputTokens")
        ),
        web_search_requests=_nonnegative_int(record.get("webSearchRequests")),
        cost_usd=_nonnegative_float(record.get("costUSD")),
        context_window=_optional_nonnegative_int(record.get("contextWindow")),
        max_output_tokens=_optional_nonnegative_int(record.get("maxOutputTokens")),
        canonical_model=_optional_string(record.get("canonicalModel")),
        provider=_optional_string(record.get("provider")),
    )


def _add_usage(left: ClaudeModelUsage, right: ClaudeModelUsage) -> ClaudeModelUsage:
    return ClaudeModelUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        cache_read_input_tokens=(
            left.cache_read_input_tokens + right.cache_read_input_tokens
        ),
        cache_creation_input_tokens=(
            left.cache_creation_input_tokens + right.cache_creation_input_tokens
        ),
        web_search_requests=left.web_search_requests + right.web_search_requests,
        cost_usd=left.cost_usd + right.cost_usd,
        context_window=_maximum_optional(left.context_window, right.context_window),
        max_output_tokens=_maximum_optional(
            left.max_output_tokens, right.max_output_tokens
        ),
        canonical_model=right.canonical_model or left.canonical_model,
        provider=right.provider or left.provider,
    )


def parse_claude_model_routing(
    trajectory_data: object,
    *,
    requested_model: str,
    switch_models_on_flag: bool,
) -> ClaudeModelRouting:
    """Extract model selection, safety fallback, and usage from Claude events."""

    if not isinstance(trajectory_data, list):
        raise TypeError("Claude Code trajectory must be a list")

    initial_model: str | None = None
    effective_model: str | None = None
    safety_fallbacks: list[ClaudeSafetyFallback] = []
    per_model_usage: dict[str, ClaudeModelUsage] = {}
    terminal_stop_reason: str | None = None
    terminal_result_succeeded = False

    for raw_event in trajectory_data:
        if not isinstance(raw_event, dict):
            continue
        event: dict[str, Any] = raw_event
        event_type = event.get("type")
        event_subtype = event.get("subtype")

        if initial_model is None and event_type == "system" and event_subtype == "init":
            initial_model = _optional_string(event.get("model"))

        if event_type == "system" and event_subtype == "model_refusal_fallback":
            original_model = _optional_string(event.get("originalModel"))
            fallback_model = _optional_string(event.get("fallbackModel"))
            if original_model is not None and fallback_model is not None:
                safety_fallbacks.append(
                    ClaudeSafetyFallback(
                        original_model=original_model,
                        fallback_model=fallback_model,
                        trigger=_optional_string(event.get("trigger")),
                        category=_optional_string(event.get("apiRefusalCategory")),
                        explanation=_optional_string(
                            event.get("apiRefusalExplanation")
                        ),
                        session_id=_optional_string(event.get("session_id")),
                    )
                )

        if event_type == "assistant" and event.get("parent_tool_use_id") is None:
            raw_message = event.get("message")
            if isinstance(raw_message, dict):
                assistant_model = _optional_string(raw_message.get("model"))
                if assistant_model is not None:
                    effective_model = assistant_model

        if event_type != "result":
            continue
        terminal_stop_reason = _optional_string(event.get("stop_reason"))
        terminal_result_succeeded = (
            event.get("subtype") == "success" and event.get("is_error") is False
        )
        raw_model_usage = event.get("modelUsage")
        if not isinstance(raw_model_usage, dict):
            continue
        for raw_model_name, raw_usage in raw_model_usage.items():
            if not isinstance(raw_model_name, str) or not isinstance(raw_usage, dict):
                continue
            current = _usage_from_record(raw_usage)
            previous = per_model_usage.get(raw_model_name)
            per_model_usage[raw_model_name] = (
                current if previous is None else _add_usage(previous, current)
            )

    if effective_model is None and safety_fallbacks:
        effective_model = safety_fallbacks[-1].fallback_model
    if effective_model is None:
        effective_model = initial_model

    return ClaudeModelRouting(
        requested_model=requested_model,
        switch_models_on_flag=switch_models_on_flag,
        initial_model=initial_model,
        effective_model=effective_model,
        fallback_occurred=bool(safety_fallbacks),
        fallback_recovered=(
            bool(safety_fallbacks)
            and terminal_result_succeeded
            and terminal_stop_reason != "refusal"
        ),
        terminal_stop_reason=terminal_stop_reason,
        safety_fallbacks=safety_fallbacks,
        per_model_usage=per_model_usage,
    )


def load_claude_model_routing(
    trajectory_path: Path,
    *,
    requested_model: str,
    switch_models_on_flag: bool,
) -> ClaudeModelRouting:
    trajectory_data: object = json.loads(trajectory_path.read_text(encoding="utf-8"))
    return parse_claude_model_routing(
        trajectory_data,
        requested_model=requested_model,
        switch_models_on_flag=switch_models_on_flag,
    )
