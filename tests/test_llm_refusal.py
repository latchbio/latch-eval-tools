import json

from latch_eval_tools.llm_refusal import detect_llm_refusal


def test_detects_anthropic_usage_policy_refusal_in_trajectory() -> None:
    result = detect_llm_refusal(
        trajectory_data={
            "messages": [
                {"text": "I am unable to respond due to Anthropic usage policy."}
            ]
        }
    )
    assert result is not None
    assert result.provider == "anthropic"
    assert result.source == "trajectory"


def test_detects_openai_content_filter_refusal() -> None:
    result = detect_llm_refusal(trajectory_data={"finish_reason": "content_filter"})
    assert result is not None
    assert result.provider == "openai"


def test_returns_none_for_normal_output() -> None:
    assert detect_llm_refusal(trajectory_data={"answer": "42 cells"}) is None


def test_sidecar_events_take_precedence() -> None:
    result = detect_llm_refusal(
        refusal_events_data=[{"provider": "anthropic", "raw_reason": "safety"}]
    )
    assert result is not None
    assert result.source == "refusal_sidecar"
    assert result.provider == "anthropic"


def test_diagnostic_bounds_large_trajectory_fields_for_agent_completion() -> None:
    hostile_text = ("\x01" * 20_000) + ('"\\🧪' * 20_000)
    result = detect_llm_refusal(
        trajectory_data={
            "code": hostile_text,
            "message": (
                f"I am unable to respond due to Anthropic usage policy. {hostile_text}"
            ),
        }
    )

    assert result is not None
    assert result.code is not None
    assert result.raw_excerpt is not None
    assert result.code.endswith("…")
    assert result.message.endswith("…")
    assert result.raw_excerpt.endswith("…")
    assert len(json.dumps(result.code, ensure_ascii=False).encode("utf-8")) <= 512
    assert len(json.dumps(result.message, ensure_ascii=False).encode("utf-8")) <= 4096
    assert (
        len(json.dumps(result.raw_excerpt, ensure_ascii=False).encode("utf-8")) <= 4096
    )

    completion_details = {
        "run_summary": {
            "schema_version": 1,
            "metrics": {
                "duration_seconds": 1.0,
                "turn_count": 1,
                "step_count": 1,
                "usage": {},
                "total_cost_usd": None,
                "cost_source": None,
                "pricing_version": None,
            },
            "refusal": {
                "status": "detected",
                "diagnostic": result.model_dump(mode="json"),
            },
        }
    }
    assert (
        len(json.dumps(completion_details, ensure_ascii=False).encode("utf-8"))
        <= 32_768
    )


def test_diagnostic_bounds_large_sidecar_and_agent_error_fields() -> None:
    hostile_text = '\x02"\\🧪' * 20_000
    sidecar_result = detect_llm_refusal(
        refusal_events_data=[
            {
                "provider": "anthropic",
                "raw_reason": hostile_text,
                "explanation": (
                    "I am unable to respond due to Anthropic usage policy. "
                    f"{hostile_text}"
                ),
            }
        ]
    )
    error_result = detect_llm_refusal(
        agent_error=(
            '{"code":"invalid_prompt","message":'
            f'"OpenAI limited access for safety reasons {hostile_text}"'
            "}"
        )
    )

    assert sidecar_result is not None
    assert sidecar_result.code is not None
    assert sidecar_result.raw_excerpt is not None
    assert sidecar_result.code.endswith("…")
    assert sidecar_result.message.endswith("…")
    assert error_result is not None
    assert error_result.message.endswith("…")
    assert error_result.raw_excerpt is not None

    serialized = json.dumps(
        {
            "sidecar": sidecar_result.model_dump(mode="json"),
            "agent_error": error_result.model_dump(mode="json"),
        },
        ensure_ascii=False,
    )
    assert len(serialized.encode("utf-8")) <= 32_768


def _claude_code_fallback_trajectory(
    *, result_event: dict[str, object]
) -> list[dict[str, object]]:
    return [
        {"type": "system", "subtype": "init", "model": "claude-opus-4-8"},
        {
            "type": "system",
            "subtype": "model_refusal_fallback",
            "originalModel": "claude-opus-4-8",
            "fallbackModel": "claude-sonnet-4-6",
            "apiRefusalCategory": "bio",
            "apiRefusalExplanation": (
                "API integrators: you can reduce refusals for your users by "
                "configuring a fallback model — see "
                "https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback"
            ),
        },
        {
            "type": "assistant",
            "message": {"model": "claude-sonnet-4-6", "stop_reason": "tool_use"},
        },
        result_event,
    ]


def test_recovered_claude_code_fallback_is_not_a_refusal() -> None:
    # Claude Code switched models after a classifier refusal and then finished the run.
    # Its model_refusal_fallback event carries Anthropic's "configure a fallback model"
    # advisory, which must not mark the completed run as refused.
    assert (
        detect_llm_refusal(
            trajectory_data=_claude_code_fallback_trajectory(
                result_event={
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "stop_reason": "end_turn",
                }
            )
        )
        is None
    )


def test_unrecovered_claude_code_fallback_is_still_a_refusal() -> None:
    # Same shape, except the fallback model refused too, so the run really did end
    # refused and the terminal stop reason says so.
    result = detect_llm_refusal(
        trajectory_data=_claude_code_fallback_trajectory(
            result_event={
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "stop_reason": "refusal",
            }
        )
    )
    assert result is not None
    assert result.provider == "anthropic"
    assert result.code == "refusal"


def test_terminal_result_stop_reason_beats_earlier_tool_use() -> None:
    result = detect_llm_refusal(
        trajectory_data=[
            {"type": "assistant", "message": {"stop_reason": "tool_use"}},
            {"type": "result", "is_error": True, "stop_reason": "refusal"},
        ]
    )
    assert result is not None
    assert result.provider == "anthropic"
    assert result.code == "refusal"


def test_anthropic_error_with_fallback_hint_is_still_a_refusal() -> None:
    # No model_refusal_fallback event: the advisory arrived on a failed request, so the
    # call was refused with nothing to recover it.
    result = detect_llm_refusal(
        trajectory_data=[
            {
                "role": "assistant",
                "model": "claude-opus-4-8",
                "stopReason": "error",
                "errorMessage": (
                    "API integrators: you can reduce refusals for your users by "
                    "configuring a fallback model — see "
                    "https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback"
                ),
            }
        ]
    )
    assert result is not None
    assert result.provider == "anthropic"
    assert result.source == "trajectory"


def test_write_refusal_verdict_writes_diagnostic(tmp_path) -> None:
    from latch_eval_tools.harness._cli_runner import (
        REFUSAL_VERDICT_FILENAME,
        _write_refusal_verdict,
    )

    _write_refusal_verdict(
        tmp_path, [{"text": "I am unable to respond due to Anthropic usage policy"}]
    )
    verdict = json.loads((tmp_path / REFUSAL_VERDICT_FILENAME).read_text())
    assert verdict["provider"] == "anthropic"
    assert verdict["source"] == "trajectory"


def test_write_refusal_verdict_writes_null_for_normal_run(tmp_path) -> None:
    from latch_eval_tools.harness._cli_runner import (
        REFUSAL_VERDICT_FILENAME,
        _write_refusal_verdict,
    )

    _write_refusal_verdict(tmp_path, [{"text": "42 cells"}])
    assert json.loads((tmp_path / REFUSAL_VERDICT_FILENAME).read_text()) is None
