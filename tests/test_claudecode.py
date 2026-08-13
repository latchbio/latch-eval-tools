import json
from pathlib import Path
from typing import Any

import pytest

from latch_eval_tools.harness import claudecode
from latch_eval_tools.harness.claude_model_routing import parse_claude_model_routing


@pytest.mark.parametrize(
    ("switch_models_on_flag", "expected_setting"),
    [(True, True), (False, False), (None, None)],
)
def test_run_claudecode_task_configures_model_switching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    switch_models_on_flag: bool | None,
    expected_setting: bool | None,
) -> None:
    observed: dict[str, Any] = {}
    expected_result = {"answer": None, "metadata": {}}

    def fake_run_cli_agent(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return expected_result

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(claudecode, "_run_cli_agent", fake_run_cli_agent)

    result = claudecode.run_claudecode_task(
        "task",
        tmp_path,
        model_name="anthropic/claude-fable-5",
        prompt_suffix="",
        switch_models_on_flag=switch_models_on_flag,
    )

    assert result is expected_result
    model_map = observed["model_map"]
    assert isinstance(model_map, dict)
    assert model_map["anthropic/claude-fable-5"] == "claude-fable-5"

    extra_args = observed["claude_code_extra_args"]
    if expected_setting is None:
        assert extra_args is None
        return
    assert isinstance(extra_args, list)
    assert extra_args[0] == "--settings"
    assert json.loads(extra_args[1]) == {
        "switchModelsOnFlag": expected_setting,
    }


def test_run_claudecode_task_rejects_non_boolean_model_switching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    with pytest.raises(TypeError, match="switch_models_on_flag"):
        claudecode.run_claudecode_task(
            "task",
            tmp_path,
            prompt_suffix="",
            switch_models_on_flag=1,  # type: ignore[arg-type]
        )


def test_model_routing_records_fallback_and_aggregates_usage() -> None:
    routing = parse_claude_model_routing(
        [
            {"type": "system", "subtype": "init", "model": "claude-fable-5"},
            {
                "type": "system",
                "subtype": "model_refusal_fallback",
                "trigger": "refusal",
                "originalModel": "claude-fable-5",
                "fallbackModel": "claude-opus-5",
                "apiRefusalCategory": "bio",
                "apiRefusalExplanation": "Biology classifier routed the request.",
                "session_id": "session-id",
            },
            {
                "type": "assistant",
                "parent_tool_use_id": None,
                "message": {"model": "claude-opus-5", "content": []},
            },
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "stop_reason": "end_turn",
                "modelUsage": {
                    "claude-fable-5": {
                        "inputTokens": 100,
                        "outputTokens": 0,
                        "costUSD": 0.01,
                    },
                    "claude-opus-5": {
                        "inputTokens": 200,
                        "outputTokens": 50,
                        "costUSD": 0.05,
                    },
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "stop_reason": "end_turn",
                "modelUsage": {
                    "claude-opus-5": {
                        "inputTokens": 300,
                        "outputTokens": 75,
                        "costUSD": 0.07,
                    }
                },
            },
        ],
        requested_model="anthropic/claude-fable-5",
        switch_models_on_flag=True,
    )

    assert routing.initial_model == "claude-fable-5"
    assert routing.effective_model == "claude-opus-5"
    assert routing.fallback_occurred
    assert routing.fallback_recovered
    assert routing.terminal_stop_reason == "end_turn"
    assert routing.safety_fallbacks[0].category == "bio"
    opus_usage = routing.per_model_usage["claude-opus-5"]
    assert opus_usage.input_tokens == 500
    assert opus_usage.output_tokens == 125
    assert opus_usage.cost_usd == pytest.approx(0.12)


def test_model_routing_preserves_terminal_fallback_refusal() -> None:
    routing = parse_claude_model_routing(
        [
            {
                "type": "system",
                "subtype": "model_refusal_fallback",
                "originalModel": "claude-fable-5",
                "fallbackModel": "claude-opus-5",
            },
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "stop_reason": "refusal",
            },
        ],
        requested_model="anthropic/claude-fable-5",
        switch_models_on_flag=True,
    )

    assert routing.fallback_occurred
    assert not routing.fallback_recovered
    assert routing.effective_model == "claude-opus-5"
