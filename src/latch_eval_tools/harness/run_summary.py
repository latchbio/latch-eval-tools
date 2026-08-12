from __future__ import annotations

import math
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from latch_eval_tools.llm_refusal import (
    LLMRefusalDiagnostic,
    detect_llm_refusal,
)

CliHarnessAgentType = Literal["claudecode", "openaicodex", "pi", "grokbuild"]
HarnessCostSource = Literal["provider_reported", "latch_eval_tools_pricing"]
HarnessRefusalStatus = Literal["detected", "not_detected", "not_evaluated"]

RUN_SUMMARY_SCHEMA_VERSION = 1
HARNESS_PRICING_VERSION = "1"
_MILLION_TOKENS = 1_000_000
_CODEX_MODEL_RATES: dict[str, dict[str, float]] = {
    "openai/gpt-5.4": {
        "input_tokens": 2.50 / _MILLION_TOKENS,
        "cache_read_tokens": 0.25 / _MILLION_TOKENS,
        "output_tokens": 15.00 / _MILLION_TOKENS,
    },
    "openai/gpt-5.5": {
        "input_tokens": 5.00 / _MILLION_TOKENS,
        "cache_read_tokens": 0.50 / _MILLION_TOKENS,
        "output_tokens": 30.00 / _MILLION_TOKENS,
    },
}
_PI_MODEL_RATES: dict[str, dict[str, float]] = {
    "anthropic/claude-opus-4-7": {
        "input_tokens": 5.00 / _MILLION_TOKENS,
        "output_tokens": 25.00 / _MILLION_TOKENS,
        "cache_read_tokens": 0.50 / _MILLION_TOKENS,
        "cache_write_tokens": 6.25 / _MILLION_TOKENS,
    },
    "anthropic/claude-opus-4-8": {
        "input_tokens": 5.00 / _MILLION_TOKENS,
        "output_tokens": 25.00 / _MILLION_TOKENS,
        "cache_read_tokens": 0.50 / _MILLION_TOKENS,
        "cache_write_tokens": 6.25 / _MILLION_TOKENS,
    },
}


class HarnessUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class HarnessRunMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    duration_seconds: float = Field(ge=0)
    turn_count: int | None = Field(default=None, ge=0)
    step_count: int | None = Field(default=None, ge=0)
    usage: HarnessUsage = Field(default_factory=HarnessUsage)
    total_cost_usd: float | None = Field(default=None, ge=0)
    cost_source: HarnessCostSource | None = None
    pricing_version: str | None = None

    @model_validator(mode="after")
    def _validate_cost_provenance(self) -> Self:
        if self.total_cost_usd is None:
            if self.cost_source is not None or self.pricing_version is not None:
                raise ValueError(
                    "cost_source and pricing_version require total_cost_usd"
                )
            return self
        if self.cost_source is None:
            raise ValueError("total_cost_usd requires cost_source")
        if (
            self.cost_source == "latch_eval_tools_pricing"
            and self.pricing_version is None
        ):
            raise ValueError("latch_eval_tools_pricing costs require pricing_version")
        return self


class HarnessRefusalAssessment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: HarnessRefusalStatus
    diagnostic: LLMRefusalDiagnostic | None = None

    @model_validator(mode="after")
    def _validate_diagnostic(self) -> Self:
        if self.status == "detected" and self.diagnostic is None:
            raise ValueError("detected refusal requires a diagnostic")
        if self.status != "detected" and self.diagnostic is not None:
            raise ValueError("only a detected refusal can include a diagnostic")
        return self


class HarnessRunSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = RUN_SUMMARY_SCHEMA_VERSION
    metrics: HarnessRunMetrics
    refusal: HarnessRefusalAssessment


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _nonnegative_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _dict_value(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return value


def _usage_from_claude_result(result: dict[str, Any] | None) -> HarnessUsage:
    if result is None:
        return HarnessUsage()
    usage = _dict_value(result.get("usage"))
    if usage is None:
        return HarnessUsage()
    return HarnessUsage(
        input_tokens=_nonnegative_int(usage.get("input_tokens")),
        output_tokens=_nonnegative_int(usage.get("output_tokens")),
        cache_read_tokens=_nonnegative_int(usage.get("cache_read_input_tokens")),
        cache_write_tokens=_nonnegative_int(usage.get("cache_creation_input_tokens")),
        reasoning_tokens=_nonnegative_int(usage.get("reasoning_tokens")),
    )


def _claude_step_count(trajectory: list[dict[str, Any]]) -> int:
    step_count = 0
    for event in trajectory:
        if event.get("type") != "assistant":
            continue
        message = _dict_value(event.get("message"))
        if message is None:
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        step_count += sum(
            1
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"
        )
    return step_count


def _claude_metrics(
    trajectory: list[dict[str, Any]], duration_seconds: float
) -> HarnessRunMetrics:
    result = next(
        (event for event in reversed(trajectory) if event.get("type") == "result"),
        None,
    )
    total_cost_usd = (
        _nonnegative_float(result.get("total_cost_usd")) if result is not None else None
    )
    return HarnessRunMetrics(
        duration_seconds=duration_seconds,
        turn_count=(
            _nonnegative_int(result.get("num_turns")) if result is not None else None
        ),
        step_count=_claude_step_count(trajectory) if trajectory else None,
        usage=_usage_from_claude_result(result),
        total_cost_usd=total_cost_usd,
        cost_source="provider_reported" if total_cost_usd is not None else None,
    )


def _codex_token_usage(
    sidecar_events: list[dict[str, Any]] | None,
) -> HarnessUsage | None:
    if sidecar_events is None:
        return None
    for event in reversed(sidecar_events):
        if event.get("type") != "event_msg":
            continue
        payload = _dict_value(event.get("payload"))
        if payload is None or payload.get("type") != "token_count":
            continue
        info = _dict_value(payload.get("info"))
        if info is None:
            continue
        usage = _dict_value(info.get("total_token_usage"))
        if usage is None:
            continue
        return HarnessUsage(
            input_tokens=_nonnegative_int(usage.get("input_tokens")),
            output_tokens=_nonnegative_int(usage.get("output_tokens")),
            cache_read_tokens=_nonnegative_int(usage.get("cached_input_tokens")),
            reasoning_tokens=_nonnegative_int(usage.get("reasoning_output_tokens")),
        )
    return None


def _codex_stream_usage(
    trajectory: list[dict[str, Any]],
) -> HarnessUsage:
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    saw_usage = False
    for event in trajectory:
        if event.get("type") != "turn.completed":
            continue
        usage = _dict_value(event.get("usage"))
        if usage is None:
            continue
        current_input = _nonnegative_int(usage.get("input_tokens"))
        current_output = _nonnegative_int(usage.get("output_tokens"))
        current_cache_read = _nonnegative_int(usage.get("cached_input_tokens"))
        if current_input is not None:
            input_tokens += current_input
            saw_usage = True
        if current_output is not None:
            output_tokens += current_output
            saw_usage = True
        if current_cache_read is not None:
            cache_read_tokens += current_cache_read
            saw_usage = True
    if not saw_usage:
        return HarnessUsage()
    return HarnessUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
    )


def _codex_step_count(
    sidecar_events: list[dict[str, Any]] | None,
) -> int | None:
    if sidecar_events is None:
        return None
    return sum(
        1
        for event in sidecar_events
        if (
            event.get("type") == "response_item"
            and isinstance(event.get("payload"), dict)
            and event["payload"].get("type") == "function_call"
        )
    )


def _codex_cost(
    model_name: str | None,
    usage: HarnessUsage,
) -> float | None:
    if model_name is None:
        return None
    normalized_model_name = (
        model_name if model_name.startswith("openai/") else f"openai/{model_name}"
    )
    rates = _CODEX_MODEL_RATES.get(normalized_model_name)
    if rates is None:
        return None
    if usage.input_tokens is None or usage.output_tokens is None:
        return None
    cache_read_tokens = usage.cache_read_tokens or 0
    uncached_input_tokens = max(usage.input_tokens - cache_read_tokens, 0)
    return (
        uncached_input_tokens * rates["input_tokens"]
        + cache_read_tokens * rates["cache_read_tokens"]
        + usage.output_tokens * rates["output_tokens"]
    )


def _codex_metrics(
    trajectory: list[dict[str, Any]],
    duration_seconds: float,
    model_name: str | None,
    sidecar_events: list[dict[str, Any]] | None,
) -> HarnessRunMetrics:
    sidecar_usage = _codex_token_usage(sidecar_events)
    usage = (
        sidecar_usage if sidecar_usage is not None else _codex_stream_usage(trajectory)
    )
    total_cost_usd = _codex_cost(model_name, usage)
    return HarnessRunMetrics(
        duration_seconds=duration_seconds,
        turn_count=(
            sum(1 for event in trajectory if event.get("type") == "turn.completed")
            if trajectory
            else None
        ),
        step_count=_codex_step_count(sidecar_events),
        usage=usage,
        total_cost_usd=total_cost_usd,
        cost_source=(
            "latch_eval_tools_pricing" if total_cost_usd is not None else None
        ),
        pricing_version=(
            HARNESS_PRICING_VERSION if total_cost_usd is not None else None
        ),
    )


def _pi_usage_and_cost(
    trajectory: list[dict[str, Any]],
) -> tuple[HarnessUsage, float | None]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }
    observed = {key: False for key in totals}
    total_cost_usd = 0.0
    saw_cost = False

    usage_fields = {
        "input_tokens": "input",
        "output_tokens": "output",
        "cache_read_tokens": "cacheRead",
        "cache_write_tokens": "cacheWrite",
        "reasoning_tokens": "reasoning",
    }
    for event in trajectory:
        if event.get("type") != "message_end":
            continue
        message = _dict_value(event.get("message"))
        if message is None or message.get("role") != "assistant":
            continue
        usage = _dict_value(message.get("usage"))
        if usage is None:
            continue
        for canonical_name, pi_name in usage_fields.items():
            value = _nonnegative_int(usage.get(pi_name))
            if value is None:
                continue
            totals[canonical_name] += value
            observed[canonical_name] = True
        cost = _dict_value(usage.get("cost"))
        if cost is None:
            continue
        current_cost = _nonnegative_float(cost.get("total"))
        if current_cost is None:
            continue
        total_cost_usd += current_cost
        saw_cost = True

    return (
        HarnessUsage(
            input_tokens=(totals["input_tokens"] if observed["input_tokens"] else None),
            output_tokens=(
                totals["output_tokens"] if observed["output_tokens"] else None
            ),
            cache_read_tokens=(
                totals["cache_read_tokens"] if observed["cache_read_tokens"] else None
            ),
            cache_write_tokens=(
                totals["cache_write_tokens"] if observed["cache_write_tokens"] else None
            ),
            reasoning_tokens=(
                totals["reasoning_tokens"] if observed["reasoning_tokens"] else None
            ),
        ),
        total_cost_usd if saw_cost else None,
    )


def _pi_step_count(trajectory: list[dict[str, Any]]) -> int:
    tool_call_ids: set[str] = set()
    anonymous_tool_calls = 0
    for event in trajectory:
        if event.get("type") != "message_end":
            continue
        message = _dict_value(event.get("message"))
        if message is None or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "toolCall":
                continue
            tool_call_id = block.get("id")
            if isinstance(tool_call_id, str) and tool_call_id:
                tool_call_ids.add(tool_call_id)
            else:
                anonymous_tool_calls += 1
    return len(tool_call_ids) + anonymous_tool_calls


def _pi_metrics(
    trajectory: list[dict[str, Any]],
    duration_seconds: float,
    model_name: str | None,
) -> HarnessRunMetrics:
    usage, total_cost_usd = _pi_usage_and_cost(trajectory)
    cost_source: HarnessCostSource | None = (
        "provider_reported" if total_cost_usd is not None else None
    )
    pricing_version = None
    rates = _PI_MODEL_RATES.get(model_name) if model_name is not None else None
    if (
        total_cost_usd is None
        and rates is not None
        and usage.input_tokens is not None
        and usage.output_tokens is not None
    ):
        total_cost_usd = (
            usage.input_tokens * rates["input_tokens"]
            + usage.output_tokens * rates["output_tokens"]
            + (usage.cache_read_tokens or 0) * rates["cache_read_tokens"]
            + (usage.cache_write_tokens or 0) * rates["cache_write_tokens"]
        )
        cost_source = "latch_eval_tools_pricing"
        pricing_version = HARNESS_PRICING_VERSION
    return HarnessRunMetrics(
        duration_seconds=duration_seconds,
        turn_count=(
            sum(1 for event in trajectory if event.get("type") == "turn_end")
            if trajectory
            else None
        ),
        step_count=_pi_step_count(trajectory) if trajectory else None,
        usage=usage,
        total_cost_usd=total_cost_usd,
        cost_source=cost_source,
        pricing_version=pricing_version,
    )


# ---------------------------------------------------------------------------
# grok-build (xai-org/grok-build) — headless `--output-format streaming-json`
#
# Schema verified against a real capture (grok 1.0.3; examples/grok_smoke_traj.jsonl).
# grok emits flat newline-delimited events keyed by a top-level `type` (NOT the
# ACP `session/update` envelope the docs implied). The ones we consume:
#   {"type":"text","data":"..."}                     assistant text chunk
#   {"type":"tool_call","toolName":"write",...}      one tool invocation
#   {"type":"tool_call_update",...}                  tool progress/result
#   {"type":"usage","usage":{...}}                   per-turn token usage
#   {"type":"end","stopReason":"end_turn",           final event (authoritative):
#     "sessionId":"...","num_turns":N,
#     "total_cost_usd":C,
#     "usage":{"input_tokens":..,"output_tokens":..,
#              "cache_read_input_tokens":..,"cache_creation_input_tokens":..,
#              "reasoning_tokens":..,"total_tokens":..}}
# The `end` event carries cumulative usage, a provider-reported dollar cost, the
# turn count, and the session id -- so cost is provider_reported (no pricing
# table needed) and resume works (sessionId is top-level).
# ---------------------------------------------------------------------------


def _grok_usage_from_dict(usage: dict[str, Any] | None) -> HarnessUsage:
    if usage is None:
        return HarnessUsage()
    return HarnessUsage(
        input_tokens=_nonnegative_int(usage.get("input_tokens")),
        output_tokens=_nonnegative_int(usage.get("output_tokens")),
        cache_read_tokens=_nonnegative_int(usage.get("cache_read_input_tokens")),
        cache_write_tokens=_nonnegative_int(usage.get("cache_creation_input_tokens")),
        reasoning_tokens=_nonnegative_int(usage.get("reasoning_tokens")),
    )


def _grok_end_event(trajectory: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (event for event in reversed(trajectory) if event.get("type") == "end"),
        None,
    )


def _grok_usage(trajectory: list[dict[str, Any]]) -> HarnessUsage:
    # Prefer the cumulative usage on the final `end` event; fall back to summing
    # the per-turn `usage` events if the run ended without one (timeout/crash).
    end = _grok_end_event(trajectory)
    if end is not None:
        usage = _dict_value(end.get("usage"))
        if usage is not None:
            return _grok_usage_from_dict(usage)

    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }
    field_map = {
        "input_tokens": "input_tokens",
        "output_tokens": "output_tokens",
        "cache_read_tokens": "cache_read_input_tokens",
        "cache_write_tokens": "cache_creation_input_tokens",
        "reasoning_tokens": "reasoning_tokens",
    }
    observed = {key: False for key in totals}
    for event in trajectory:
        if event.get("type") != "usage":
            continue
        usage = _dict_value(event.get("usage"))
        if usage is None:
            continue
        for canonical, grok_name in field_map.items():
            value = _nonnegative_int(usage.get(grok_name))
            if value is not None:
                totals[canonical] += value
                observed[canonical] = True
    return HarnessUsage(
        **{key: (totals[key] if observed[key] else None) for key in totals}
    )


def _grok_step_count(trajectory: list[dict[str, Any]]) -> int:
    # One step per tool invocation (`tool_call`); `tool_call_update` events are
    # progress on an already-counted call and are not counted.
    return sum(1 for event in trajectory if event.get("type") == "tool_call")


def _grok_metrics(
    trajectory: list[dict[str, Any]],
    duration_seconds: float,
    model_name: str | None,
) -> HarnessRunMetrics:
    end = _grok_end_event(trajectory)
    total_cost_usd = (
        _nonnegative_float(end.get("total_cost_usd")) if end is not None else None
    )
    turn_count = _nonnegative_int(end.get("num_turns")) if end is not None else None
    return HarnessRunMetrics(
        duration_seconds=duration_seconds,
        turn_count=turn_count,
        step_count=_grok_step_count(trajectory) if trajectory else None,
        usage=_grok_usage(trajectory),
        total_cost_usd=total_cost_usd,
        cost_source="provider_reported" if total_cost_usd is not None else None,
    )


def _miniswe_messages(
    serialized_trajectory: dict[str, Any],
) -> list[dict[str, Any]] | None:
    raw_messages = serialized_trajectory.get("messages")
    if not isinstance(raw_messages, list):
        return None
    return [message for message in raw_messages if isinstance(message, dict)]


def _miniswe_usage(
    messages: list[dict[str, Any]] | None,
) -> HarnessUsage:
    if messages is None:
        return HarnessUsage()

    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }
    observed = {key: False for key in totals}
    for message in messages:
        if message.get("role") != "assistant" and message.get("object") != "response":
            continue
        usage = _dict_value(message.get("usage"))
        if usage is None:
            extra = _dict_value(message.get("extra"))
            response = _dict_value(extra.get("response")) if extra is not None else None
            usage = _dict_value(response.get("usage")) if response is not None else None
            if usage is None:
                continue

        input_tokens = _nonnegative_int(usage.get("input_tokens"))
        if input_tokens is None:
            input_tokens = _nonnegative_int(usage.get("prompt_tokens"))
        if input_tokens is not None:
            totals["input_tokens"] += input_tokens
            observed["input_tokens"] = True

        output_tokens = _nonnegative_int(usage.get("output_tokens"))
        if output_tokens is None:
            output_tokens = _nonnegative_int(usage.get("completion_tokens"))
        if output_tokens is not None:
            totals["output_tokens"] += output_tokens
            observed["output_tokens"] = True

        input_details = _dict_value(usage.get("input_tokens_details"))
        if input_details is None:
            input_details = _dict_value(usage.get("prompt_tokens_details"))
        cache_read_tokens = _nonnegative_int(usage.get("cache_read_input_tokens"))
        if cache_read_tokens is None and input_details is not None:
            cache_read_tokens = _nonnegative_int(input_details.get("cached_tokens"))
        if cache_read_tokens is not None:
            totals["cache_read_tokens"] += cache_read_tokens
            observed["cache_read_tokens"] = True

        cache_write_tokens = _nonnegative_int(usage.get("cache_creation_input_tokens"))
        if cache_write_tokens is not None:
            totals["cache_write_tokens"] += cache_write_tokens
            observed["cache_write_tokens"] = True

        output_details = _dict_value(usage.get("output_tokens_details"))
        if output_details is None:
            output_details = _dict_value(usage.get("completion_tokens_details"))
        reasoning_tokens = _nonnegative_int(usage.get("reasoning_tokens"))
        if reasoning_tokens is None and output_details is not None:
            reasoning_tokens = _nonnegative_int(output_details.get("reasoning_tokens"))
        if reasoning_tokens is not None:
            totals["reasoning_tokens"] += reasoning_tokens
            observed["reasoning_tokens"] = True

    return HarnessUsage(
        input_tokens=(totals["input_tokens"] if observed["input_tokens"] else None),
        output_tokens=(totals["output_tokens"] if observed["output_tokens"] else None),
        cache_read_tokens=(
            totals["cache_read_tokens"] if observed["cache_read_tokens"] else None
        ),
        cache_write_tokens=(
            totals["cache_write_tokens"] if observed["cache_write_tokens"] else None
        ),
        reasoning_tokens=(
            totals["reasoning_tokens"] if observed["reasoning_tokens"] else None
        ),
    )


def assess_llm_refusal(
    *,
    trajectory: list[dict[str, Any]] | dict[str, Any],
    refusal_events: list[dict[str, Any]] | None = None,
    agent_error: str | None = None,
) -> HarnessRefusalAssessment:
    if isinstance(trajectory, dict):
        raw_messages = trajectory.get("messages")
        trajectory_evidence_available = isinstance(raw_messages, list) and any(
            isinstance(message, dict) and bool(message) for message in raw_messages
        )
    else:
        trajectory_evidence_available = bool(trajectory)
    evidence_available = (
        trajectory_evidence_available
        or refusal_events is not None
        or agent_error is not None
    )
    if not evidence_available:
        return HarnessRefusalAssessment(
            status="not_evaluated",
            diagnostic=None,
        )

    diagnostic = detect_llm_refusal(
        trajectory_data=trajectory,
        agent_error=agent_error,
        refusal_events_data=refusal_events,
    )
    if diagnostic is None:
        return HarnessRefusalAssessment(
            status="not_detected",
            diagnostic=None,
        )
    return HarnessRefusalAssessment(
        status="detected",
        diagnostic=diagnostic,
    )


def build_miniswe_run_summary(
    *,
    serialized_trajectory: dict[str, Any],
    duration_seconds: float,
    total_cost_usd: object,
    step_count: object,
    agent_error: str | None = None,
) -> HarnessRunSummary:
    normalized_duration = _nonnegative_float(duration_seconds)
    if normalized_duration is None:
        raise ValueError("duration_seconds must be a finite nonnegative number")

    messages = _miniswe_messages(serialized_trajectory)
    normalized_cost = _nonnegative_float(total_cost_usd)
    normalized_step_count = _nonnegative_int(step_count)
    return HarnessRunSummary(
        metrics=HarnessRunMetrics(
            duration_seconds=normalized_duration,
            turn_count=normalized_step_count,
            step_count=normalized_step_count,
            usage=_miniswe_usage(messages),
            total_cost_usd=normalized_cost,
            cost_source=("provider_reported" if normalized_cost is not None else None),
        ),
        refusal=assess_llm_refusal(
            trajectory=serialized_trajectory,
            agent_error=agent_error,
        ),
    )


def build_cli_run_summary(
    *,
    agent_type: CliHarnessAgentType,
    trajectory: list[dict[str, Any]],
    duration_seconds: float,
    model_name: str | None,
    codex_sidecar_events: list[dict[str, Any]] | None = None,
    refusal_events: list[dict[str, Any]] | None = None,
    agent_error: str | None = None,
) -> HarnessRunSummary:
    normalized_duration = _nonnegative_float(duration_seconds)
    if normalized_duration is None:
        raise ValueError("duration_seconds must be a finite nonnegative number")

    if agent_type == "claudecode":
        metrics = _claude_metrics(trajectory, normalized_duration)
    elif agent_type == "openaicodex":
        metrics = _codex_metrics(
            trajectory,
            normalized_duration,
            model_name,
            codex_sidecar_events,
        )
    elif agent_type == "pi":
        metrics = _pi_metrics(
            trajectory,
            normalized_duration,
            model_name,
        )
    elif agent_type == "grokbuild":
        metrics = _grok_metrics(
            trajectory,
            normalized_duration,
            model_name,
        )
    else:
        raise ValueError(f"Unknown CLI agent type: {agent_type}")

    return HarnessRunSummary(
        metrics=metrics,
        refusal=assess_llm_refusal(
            trajectory=trajectory,
            refusal_events=refusal_events,
            agent_error=agent_error,
        ),
    )
