import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


LLMRefusalProvider = Literal["openai", "anthropic", "google", "unknown"]
LLMRefusalSource = Literal[
    "workflow_error", "trajectory", "agent_output", "refusal_sidecar"
]

# Refusal diagnostics are included in the bounded V2 agent completion. Budget the
# three free-form fields against their JSON representation so quotes, control
# characters, and multibyte text cannot unexpectedly dominate the completion.
_DIAGNOSTIC_CODE_JSON_BYTES = 512
_DIAGNOSTIC_MESSAGE_JSON_BYTES = 4096
_DIAGNOSTIC_RAW_EXCERPT_JSON_BYTES = 4096
_TRUNCATION_MARKER = "…"


_SIDECAR_PROVIDER_MAP: dict[str, LLMRefusalProvider] = {
    "anthropic": "anthropic",
    "openai-completions": "openai",
    "openai": "openai",
    "openai-responses": "openai",
    "google": "google",
}


def _json_string_size_bytes(value: str) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _truncate_json_string(value: str, *, max_bytes: int) -> str:
    value = value.replace("\x00", "\ufffd")
    if _json_string_size_bytes(value) <= max_bytes:
        return value

    # Search by Unicode code point while measuring the encoded JSON string. This
    # keeps the result valid UTF-8 and accounts for JSON escaping expansion.
    low = 0
    high = len(value)
    while low < high:
        midpoint = (low + high + 1) // 2
        candidate = f"{value[:midpoint]}{_TRUNCATION_MARKER}"
        if _json_string_size_bytes(candidate) <= max_bytes:
            low = midpoint
        else:
            high = midpoint - 1
    return f"{value[:low]}{_TRUNCATION_MARKER}"


class LLMRefusalDiagnostic(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["llm_refusal"] = "llm_refusal"
    provider: LLMRefusalProvider
    code: str | None = None
    message: str
    source: LLMRefusalSource
    raw_excerpt: str | None = None

    @field_validator("code")
    @classmethod
    def _bound_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _truncate_json_string(value, max_bytes=_DIAGNOSTIC_CODE_JSON_BYTES)

    @field_validator("message")
    @classmethod
    def _bound_message(cls, value: str) -> str:
        return _truncate_json_string(value, max_bytes=_DIAGNOSTIC_MESSAGE_JSON_BYTES)

    @field_validator("raw_excerpt")
    @classmethod
    def _bound_raw_excerpt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _truncate_json_string(
            value,
            max_bytes=_DIAGNOSTIC_RAW_EXCERPT_JSON_BYTES,
        )


def _parse_json_record(value: str | None) -> dict[str, Any] | None:
    if value is None or value == "":
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _collect_strings(value: Any, out: list[str], *, limit: int = 5000) -> None:
    if len(out) >= limit:
        return
    if isinstance(value, str):
        if value:
            out.append(value)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, out, limit=limit)
            if len(out) >= limit:
                return
        return
    if isinstance(value, list):
        for item in value:
            _collect_strings(item, out, limit=limit)
            if len(out) >= limit:
                return


def _find_string_field(value: Any, field_names: set[str]) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in field_names and isinstance(item, str) and item:
                return item
        for item in value.values():
            found = _find_string_field(item, field_names)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_string_field(item, field_names)
            if found is not None:
                return found
    return None


_ANTHROPIC_FALLBACK_MARKERS: tuple[str, ...] = (
    "refusals-and-fallback",
    "configuring a fallback model",
    "reduce refusals for your users",
)


def _find_message(strings: list[str], provider: LLMRefusalProvider) -> str:
    provider_markers: tuple[str, ...]
    if provider == "openai":
        provider_markers = (
            "invalid prompt",
            "limited access",
            "safety reasons",
            "cyber policy",
            "content_filter",
        )
    elif provider == "anthropic":
        provider_markers = (
            "unable to respond",
            "usage policy",
            "violat",
        ) + _ANTHROPIC_FALLBACK_MARKERS
    else:
        provider_markers = (
            "refusal",
            "unable to respond",
            "invalid prompt",
            "usage policy",
        )

    for item in strings:
        lowered = item.lower()
        if any(marker in lowered for marker in provider_markers):
            return item
    for item in strings:
        if item.strip():
            return item
    return "The model refused to respond to this request."


def _excerpt(strings: list[str]) -> str | None:
    joined = "\n".join(item for item in strings if item.strip())
    if joined == "":
        return None
    return joined[:1000]


def _diagnostic_from_sidecar(record: dict[str, Any]) -> LLMRefusalDiagnostic:
    raw_provider = record.get("provider")
    provider = (
        _SIDECAR_PROVIDER_MAP.get(raw_provider, "unknown")
        if isinstance(raw_provider, str)
        else "unknown"
    )
    raw_reason = record.get("raw_reason")
    explanation = record.get("explanation")
    message = (
        explanation
        if isinstance(explanation, str) and explanation
        else f"Model refusal ({raw_reason or 'unknown reason'})"
    )
    return LLMRefusalDiagnostic(
        provider=provider,
        code=raw_reason if isinstance(raw_reason, str) else None,
        message=message,
        source="refusal_sidecar",
        raw_excerpt=json.dumps(record)[:1000],
    )


def _detect_from_value(
    value: Any, *, source: LLMRefusalSource
) -> LLMRefusalDiagnostic | None:
    if value is None:
        return None

    strings: list[str] = []
    _collect_strings(value, strings)
    if not strings:
        return None

    lowered = "\n".join(strings).lower()
    code = _find_string_field(value, {"code"})
    stop_reason = _find_string_field(value, {"stop_reason", "stopReason"})
    finish_reason = _find_string_field(value, {"finish_reason", "finishReason"})

    anthropic_fallback_hit = any(
        marker in lowered for marker in _ANTHROPIC_FALLBACK_MARKERS
    )

    if (
        stop_reason in {"refusal", "sensitive"}
        or (
            "usage policy" in lowered
            and (
                "claude" in lowered
                or "anthropic" in lowered
                or "unable to respond" in lowered
            )
        )
        or anthropic_fallback_hit
    ):
        return LLMRefusalDiagnostic(
            provider="anthropic",
            code=code
            or (stop_reason if stop_reason not in {None, "error"} else None)
            or "refusal",
            message=_find_message(strings, "anthropic"),
            source=source,
            raw_excerpt=_excerpt(strings),
        )

    if (
        code == "cyber_policy"
        or "cyber policy" in lowered
        or finish_reason == "content_filter"
    ):
        return LLMRefusalDiagnostic(
            provider="openai",
            code=code or finish_reason or "cyber_policy",
            message=_find_message(strings, "openai"),
            source=source,
            raw_excerpt=_excerpt(strings),
        )

    if (
        "invalid_prompt" in lowered
        or (
            "invalid prompt" in lowered
            and ("openai" in lowered or "limited access" in lowered)
        )
        or ("limited access to this content" in lowered and "safety reasons" in lowered)
    ):
        return LLMRefusalDiagnostic(
            provider="openai",
            code=code or "invalid_prompt",
            message=_find_message(strings, "openai"),
            source=source,
            raw_excerpt=_excerpt(strings),
        )

    if "refusal" in lowered and ("policy" in lowered or "safety" in lowered):
        return LLMRefusalDiagnostic(
            provider="unknown",
            code=code,
            message=_find_message(strings, "unknown"),
            source=source,
            raw_excerpt=_excerpt(strings),
        )

    return None


def detect_llm_refusal(
    *,
    trajectory_data: Any | None = None,
    workflow_error_data: str | None = None,
    agent_error: str | None = None,
    agent_output_data: Any | None = None,
    refusal_events_data: list[dict[str, Any]] | None = None,
) -> LLMRefusalDiagnostic | None:
    if refusal_events_data:
        first = refusal_events_data[0]
        if isinstance(first, dict):
            return _diagnostic_from_sidecar(first)

    trajectory_refusal = _detect_from_value(trajectory_data, source="trajectory")
    if trajectory_refusal is not None:
        return trajectory_refusal

    workflow_error = _parse_json_record(workflow_error_data)
    workflow_refusal = _detect_from_value(
        workflow_error if workflow_error is not None else workflow_error_data,
        source="workflow_error",
    )
    if workflow_refusal is not None:
        return workflow_refusal

    agent_output_refusal = _detect_from_value(agent_output_data, source="agent_output")
    if agent_output_refusal is not None:
        return agent_output_refusal

    return _detect_from_value(agent_error, source="agent_output")
