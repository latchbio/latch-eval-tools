import json

import pytest

from latch_eval_tools.harness import _cli_runner


def test_classifies_terminal_claude_rate_limit_with_retry_hint() -> None:
    failure = _cli_runner.classify_terminal_provider_failure(
        "claudecode",
        [
            {
                "type": "system",
                "subtype": "api_retry",
                "error_status": 429,
                "error": "rate_limit",
                "retry_delay_ms": 11_000,
            },
            {
                "type": "result",
                "terminal_reason": "api_error",
                "api_error_status": 429,
                "result": "API Error: rate limit exceeded",
            },
        ],
    )

    assert failure == _cli_runner.ProviderFailure(
        status_code=429,
        retry_after_seconds=11.0,
    )
    assert failure.error_code == "rate_limit"
    assert failure.retryable
    assert failure.capacity_limited


def test_classifies_terminal_claude_overload_without_retry_hint() -> None:
    failure = _cli_runner.classify_terminal_provider_failure(
        "claudecode",
        [
            {
                "type": "result",
                "terminal_reason": "api_error",
                "api_error_status": 529,
                "result": "API Error: overloaded",
            }
        ],
    )

    assert failure == _cli_runner.ProviderFailure(
        status_code=529,
        retry_after_seconds=None,
    )
    assert failure.error_code == "overloaded"


def test_terminal_claude_failure_keeps_provider_message() -> None:
    failure = _cli_runner.classify_terminal_provider_failure(
        "claudecode",
        [
            {
                "type": "result",
                "terminal_reason": "api_error",
                "api_error_status": 400,
                "result": (
                    "API Error: 400 You have reached your specified API usage"
                    " limits. You will regain access on 2026-09-01 at 00:00 UTC."
                ),
            }
        ],
    )

    assert failure is not None
    assert failure.message is not None
    assert "specified API usage limits" in failure.message
    assert len(failure.message) <= _cli_runner.PROVIDER_MESSAGE_MAX_CHARS


def test_terminal_claude_failure_without_message_text() -> None:
    failure = _cli_runner.classify_terminal_provider_failure(
        "claudecode",
        [
            {
                "type": "result",
                "terminal_reason": "api_error",
                "api_error_status": 400,
                "result": "   ",
            }
        ],
    )

    assert failure is not None
    assert failure.message is None


def test_inflight_claude_retry_keeps_provider_message() -> None:
    failure = _cli_runner.classify_terminal_provider_failure(
        "claudecode",
        [
            {
                "type": "system",
                "subtype": "api_retry",
                "error_status": 429,
                "error": "rate_limit",
                "retry_delay_ms": 11_000,
            }
        ],
        include_inflight_retry=True,
    )

    assert failure is not None
    assert failure.message == "rate_limit"


def test_ignores_recovered_claude_api_retry() -> None:
    failure = _cli_runner.classify_terminal_provider_failure(
        "claudecode",
        [
            {
                "type": "system",
                "subtype": "api_retry",
                "error_status": 520,
                "error": "server_error",
                "retry_delay_ms": 4_000,
            },
            {
                "type": "result",
                "terminal_reason": "end_turn",
                "is_error": False,
            },
        ],
    )

    assert failure is None


def test_does_not_reuse_claude_retry_hint_across_assistant_boundary() -> None:
    failure = _cli_runner.classify_terminal_provider_failure(
        "claudecode",
        [
            {
                "type": "system",
                "subtype": "api_retry",
                "error_status": 429,
                "retry_delay_ms": 11_000,
            },
            {"type": "assistant", "message": {"role": "assistant", "content": []}},
            {
                "type": "result",
                "terminal_reason": "api_error",
                "api_error_status": 429,
            },
        ],
    )

    assert failure == _cli_runner.ProviderFailure(
        status_code=429,
        retry_after_seconds=None,
    )


def test_classifies_claude_retry_still_inflight_at_timeout() -> None:
    events = [
        {
            "type": "system",
            "subtype": "api_retry",
            "error_status": 520,
            "error": "server_error",
            "retry_delay_ms": 4_000,
        }
    ]

    assert _cli_runner.classify_terminal_provider_failure("claudecode", events) is None
    assert _cli_runner.classify_terminal_provider_failure(
        "claudecode",
        events,
        include_inflight_retry=True,
    ) == _cli_runner.ProviderFailure(
        status_code=520,
        retry_after_seconds=4.0,
    )


def test_later_claude_assistant_message_suppresses_inflight_retry() -> None:
    failure = _cli_runner.classify_terminal_provider_failure(
        "claudecode",
        [
            {
                "type": "system",
                "subtype": "api_retry",
                "error_status": 520,
                "error": "server_error",
                "retry_delay_ms": 4_000,
            },
            {"type": "assistant", "message": {"role": "assistant", "content": []}},
        ],
        include_inflight_retry=True,
    )

    assert failure is None


def test_classifies_terminal_gemini_resource_exhaustion() -> None:
    google_payload = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "27s",
                }
            ],
        }
    }
    provider_payload = {
        "error": {
            "message": json.dumps(google_payload),
            "code": 429,
            "status": "Too Many Requests",
        }
    }
    failure = _cli_runner.classify_terminal_provider_failure(
        "pi",
        [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "api": "google-generative-ai",
                    "provider": "google",
                    "model": "gemini-3.1-pro-preview",
                    "stopReason": "error",
                    "errorMessage": json.dumps(provider_payload),
                },
            }
        ],
    )

    assert failure == _cli_runner.ProviderFailure(
        status_code=429,
        retry_after_seconds=27.0,
    )


def test_classifies_terminal_openrouter_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_cli_runner.time, "time", lambda: 1_000.0)
    provider_payload = {
        "message": "Provider returned error",
        "code": 429,
        "metadata": {
            "retry_after_seconds": 11,
            "headers": {
                "Retry-After": "12",
                "X-RateLimit-Reset": "1013000",
            },
        },
    }
    failure = _cli_runner.classify_terminal_provider_failure(
        "pi",
        [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "api": "openai-completions",
                    "provider": "openrouter",
                    "model": "moonshotai/kimi-k3",
                    "stopReason": "error",
                    "errorMessage": f"429: {json.dumps(provider_payload)}",
                },
            }
        ],
    )

    assert failure == _cli_runner.ProviderFailure(
        status_code=429,
        retry_after_seconds=13.0,
    )


def test_status_wins_over_fireworks_error_type() -> None:
    provider_payload = {
        "error": {
            "type": "invalid_request_error",
            "message": "rate limit exceeded, please try again later",
        },
        "type": "error",
    }
    failure = _cli_runner.classify_terminal_provider_failure(
        "pi",
        [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "api": "anthropic-messages",
                    "provider": "fireworks",
                    "model": "accounts/fireworks/models/kimi-k2p6",
                    "stopReason": "error",
                    "errorMessage": f"429 {json.dumps(provider_payload)}",
                },
            }
        ],
    )

    assert failure == _cli_runner.ProviderFailure(
        status_code=429,
        retry_after_seconds=None,
    )
    assert failure.error_code == "rate_limit"


def test_latest_pi_assistant_message_is_authoritative() -> None:
    failure = _cli_runner.classify_terminal_provider_failure(
        "pi",
        [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": "error",
                    "errorMessage": '429: {"code": 429}',
                },
            },
            {"type": "agent_end", "willRetry": True},
            {"type": "auto_retry_start", "attempt": 2, "delayMs": 2_000},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": "toolUse",
                    "content": [],
                },
            },
            {"type": "auto_retry_end", "success": True, "attempt": 2},
        ],
    )

    assert failure is None


def test_preserves_permanent_openrouter_failure_without_retrying() -> None:
    failure = _cli_runner.classify_terminal_provider_failure(
        "pi",
        [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "provider": "openrouter",
                    "stopReason": "error",
                    "errorMessage": (
                        '402: {"message":"Insufficient credits","code":402}'
                    ),
                },
            }
        ],
    )

    assert failure == _cli_runner.ProviderFailure(
        status_code=402,
        retry_after_seconds=None,
    )
    assert failure.error_code == "http_402"
    assert not failure.retryable


def test_classifies_permanent_gemini_failure_by_status() -> None:
    google_payload = {
        "error": {
            "code": 400,
            "status": "INVALID_ARGUMENT",
            "message": "invalid request",
        }
    }
    provider_payload = {
        "error": {
            "message": json.dumps(google_payload),
            "code": 400,
            "status": "Bad Request",
        }
    }
    failure = _cli_runner.classify_terminal_provider_failure(
        "pi",
        [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "provider": "google",
                    "stopReason": "error",
                    "errorMessage": json.dumps(provider_payload),
                },
            }
        ],
    )

    assert failure == _cli_runner.ProviderFailure(
        status_code=400,
        retry_after_seconds=None,
    )
    assert failure.error_code == "http_400"
    assert not failure.retryable


@pytest.mark.parametrize(
    "attempt_events",
    [
        [{"type": "tool_result", "status": 429, "error": "rate limit"}],
        [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": "error",
                    "errorMessage": "rate limit exceeded",
                },
            }
        ],
        [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": "error",
                    "errorMessage": "429: []",
                },
            }
        ],
    ],
)
def test_ignores_unstructured_provider_evidence(
    attempt_events: list[dict],
) -> None:
    assert _cli_runner.classify_terminal_provider_failure("pi", attempt_events) is None


def test_retry_delay_honors_typed_hint_with_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _cli_runner.random,
        "uniform",
        lambda start, end: (start + end) / 2.0,
    )
    failure = _cli_runner.ProviderFailure(
        status_code=429,
        retry_after_seconds=11.0,
    )

    assert _cli_runner.provider_retry_delay_seconds(failure, 1) == 13.5


def test_capacity_retry_without_hint_uses_conservative_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_cli_runner.random, "uniform", lambda _start, end: end)
    failure = _cli_runner.ProviderFailure(
        status_code=429,
        retry_after_seconds=None,
    )

    assert _cli_runner.provider_retry_delay_seconds(failure, 1) == 75.0
    assert _cli_runner.PROVIDER_MAX_RESUMES == 5


def test_retry_delay_backs_off_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_cli_runner.random, "uniform", lambda _start, _end: 0.0)
    failure = _cli_runner.ProviderFailure(
        status_code=429,
        retry_after_seconds=45.0,
    )

    assert _cli_runner.provider_retry_delay_seconds(failure, 2) == 90.0
    assert _cli_runner.provider_retry_delay_seconds(failure, 3) == 180.0
    assert _cli_runner.provider_retry_delay_seconds(failure, 5) == 300.0


def test_claude_resume_identifier_is_only_passed_to_resume_flag() -> None:
    command = _cli_runner._build_agent_command(
        "claudecode",
        ["claude"],
        "anthropic/claude-fable-5",
        {"anthropic/claude-fable-5": "claude-fable-5"},
        ["--settings", '{"switchModelsOnFlag":true}'],
        resume_identifier="session-id",
    )

    assert command.count("session-id") == 1
    assert command[command.index("--resume") + 1] == "session-id"
    assert "--model" not in command
    settings = json.loads(command[command.index("--settings") + 1])
    assert settings == {"switchModelsOnFlag": True}


def test_claude_initial_command_selects_mapped_fable_model() -> None:
    command = _cli_runner._build_agent_command(
        "claudecode",
        ["claude"],
        "anthropic/claude-fable-5",
        {"anthropic/claude-fable-5": "claude-fable-5"},
        ["--settings", '{"switchModelsOnFlag":true}'],
    )

    assert command[command.index("--model") + 1] == "claude-fable-5"
    settings = json.loads(command[command.index("--settings") + 1])
    assert settings == {"switchModelsOnFlag": True}
