import asyncio
import json
import types

import pytest
from pydantic import ValidationError

from latch_eval_tools.graders import LLM_GRADER_REGISTRY
from latch_eval_tools.graders.rubric import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_PARAMS,
    DEFAULT_TRUNCATION_LENGTH,
    RUBRIC_GRADER_SYSTEM_PROMPT,
    RubricCriterionGraderOutput,
    RubricGrader,
    RubricGraderConfig,
    RubricGraderOutput,
    RubricGraderOutputParseError,
    build_rubric_messages,
    compute_rubric_reward,
    parse_rubric_grader_output,
    resolve_anthropic_model_name,
    rubric_criterion_output_config,
    rubric_output_config,
)


def test_rubric_config_defaults_and_requires_positive_score_delta() -> None:
    config = RubricGraderConfig.model_validate(
        {
            "answer_field": "not_enough_info.rationale",
            "criteria": [{"description": "mentions hERG", "score_delta": 1}],
        }
    )

    assert config.model_id == DEFAULT_MODEL_ID
    assert config.model_params == DEFAULT_MODEL_PARAMS
    assert config.truncation_length == DEFAULT_TRUNCATION_LENGTH
    assert config.passing_reward_threshold == 1.0

    try:
        RubricGraderConfig.model_validate(
            {
                "answer_field": "answer",
                "criteria": [{"description": "only penalty", "score_delta": -1}],
            }
        )
    except ValidationError as exc:
        assert "max_score must be > 0" in str(exc)
    else:
        raise AssertionError("expected negative-only rubric config to fail validation")


def test_compute_rubric_reward_normalizes_and_clamps() -> None:
    config = RubricGraderConfig.model_validate(
        {
            "answer_field": "rationale",
            "criteria": [
                {"description": "states that hERG is required", "score_delta": 1},
                {"description": "includes a second correct reason", "score_delta": 1},
                {"description": "claims CNS effects block the conclusion", "score_delta": -0.5},
            ],
            "passing_reward_threshold": 0.75,
        }
    )
    output = RubricGraderOutput.model_validate(
        {
            "judgments": [
                {"index": 0, "met": True, "rationale": "present"},
                {"index": 1, "met": False, "rationale": "missing"},
                {"index": 2, "met": True, "rationale": "present"},
            ]
        }
    )

    result = compute_rubric_reward(config, output)

    assert result.raw_score == 0.5
    assert result.max_score == 2
    assert result.reward == 0.25
    assert result.passed is False
    assert result.field_scores == {
        "criterion_0": 1.0,
        "criterion_1": 0.0,
        "criterion_2": -0.5,
    }


def test_rubric_prompt_and_llm_registry() -> None:
    config = RubricGraderConfig.model_validate(
        {
            "answer_field": "rationale",
            "criteria": [{"description": "states that hERG is required", "score_delta": 1}],
        }
    )

    messages = build_rubric_messages("hERG is required.", config)

    assert LLM_GRADER_REGISTRY == {"rubric": RubricGrader}
    assert messages[0] == {"role": "system", "content": RUBRIC_GRADER_SYSTEM_PROMPT}
    assert "<response>\nhERG is required.\n</response>" in messages[1]["content"]
    assert "[0] states that hERG is required" in messages[1]["content"]
    assert "Return one judgment per rubric item" in messages[1]["content"]
    assert "Keep each rationale under 256 characters." in messages[1]["content"]


def test_rubric_output_preserves_rationale() -> None:
    output = RubricGraderOutput.model_validate(
        {"judgments": [{"index": 0, "met": True, "rationale": "x" * 300}]}
    )

    assert len(output.judgments[0].rationale) == 300


def test_parse_rubric_grader_output_accepts_canonical_output() -> None:
    judgment = {"index": 0, "met": True, "rationale": "present"}
    payloads = [
        {"judgments": [judgment]},
        json.dumps({"judgments": [judgment]}),
        RubricGraderOutput.model_validate({"judgments": [judgment]}),
    ]

    for payload in payloads:
        output = parse_rubric_grader_output(payload)

        assert output.judgments[0].index == 0
        assert output.judgments[0].met is True
        assert output.judgments[0].rationale == "present"


def test_parse_rubric_grader_output_rejects_noncanonical_output() -> None:
    judgment = {"index": 0, "met": True, "rationale": "present"}
    payloads = [
        json.dumps({"judgments": json.dumps([judgment])}),
        json.dumps({"json_value": {"judgments": [judgment]}}),
        json.dumps([judgment]),
        "```json\n" + json.dumps({"judgments": [judgment]}) + "\n```",
        [{"type": "text", "text": json.dumps({"judgments": [judgment]})}],
    ]

    for payload in payloads:
        with pytest.raises((ValidationError, ValueError)):
            parse_rubric_grader_output(payload)


def test_rubric_output_rejects_empty_judgments() -> None:
    with pytest.raises(ValidationError):
        RubricGraderOutput.model_validate({"judgments": []})

    with pytest.raises(ValidationError):
        parse_rubric_grader_output(json.dumps({"judgments": []}))


def _fake_message(content: str, *, stop_reason: str = "end_turn") -> object:
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text=content)],
        stop_reason=stop_reason,
    )


class FakeTransientAnthropicError(Exception):
    pass


def _patch_no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("latch_eval_tools.graders.rubric.asyncio.sleep", fake_sleep)


def _patch_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[object],
    observed_requests: list[dict[str, object]] | None = None,
    observed_clients: list[dict[str, object]] | None = None,
) -> None:
    observed_requests = [] if observed_requests is None else observed_requests
    observed_clients = [] if observed_clients is None else observed_clients
    pending = list(messages)

    class FakeMessages:
        async def create(self, **kwargs: object) -> object:
            observed_requests.append(kwargs)
            item = pending.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs: object) -> None:
            observed_clients.append(kwargs)
            self.messages = FakeMessages()

    monkeypatch.setattr(
        "latch_eval_tools.graders.rubric.AsyncAnthropic",
        FakeAsyncAnthropic,
    )
    _patch_no_sleep(monkeypatch)


def _run_rubric_grader(monkeypatch: pytest.MonkeyPatch, message: object) -> None:
    _patch_anthropic(monkeypatch, [message, message, message, message])
    config = {
        "answer_field": "rationale",
        "criteria": [{"description": "states that hERG is required", "score_delta": 1}],
    }
    asyncio.run(RubricGrader().evaluate_answer_llm({"rationale": "hi"}, config, api_key="k"))


def test_evaluate_answer_llm_wraps_parse_failure_with_raw_content(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_content = json.dumps({"totally_unexpected": {"nope": 1}})

    with pytest.raises(RubricGraderOutputParseError) as exc_info:
        _run_rubric_grader(monkeypatch, _fake_message(raw_content))

    exc = exc_info.value
    assert exc.raw_content == raw_content
    assert exc.finish_reason == "end_turn"
    assert exc.had_tool_calls is False
    assert "totally_unexpected" in str(exc)


def test_evaluate_answer_llm_reports_refusal_stop_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RubricGraderOutputParseError) as exc_info:
        _run_rubric_grader(
            monkeypatch,
            _fake_message("I can't help with that.", stop_reason="refusal"),
        )

    exc = exc_info.value
    assert exc.raw_content == "I can't help with that."
    assert exc.finish_reason == "refusal"
    assert exc.had_tool_calls is False


def test_evaluate_answer_llm_uses_anthropic_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    judgment = json.dumps({"met": True, "rationale": "present"})
    observed_requests: list[dict[str, object]] = []
    observed_clients: list[dict[str, object]] = []
    _patch_anthropic(
        monkeypatch,
        [_fake_message(judgment)],
        observed_requests=observed_requests,
        observed_clients=observed_clients,
    )
    config = {
        "answer_field": "rationale",
        "criteria": [{"description": "states that hERG is required", "score_delta": 1}],
    }
    result = asyncio.run(RubricGrader().evaluate_answer_llm({"rationale": "hi"}, config, api_key="k"))

    assert result.passed is True
    assert result.metrics["judgments"][0]["met"] is True
    assert result.metrics["grading_transport"] == "anthropic_api"
    assert observed_clients == [{"api_key": "k"}]
    assert observed_requests[0]["model"] == "claude-sonnet-5"
    assert observed_requests[0]["system"] == RUBRIC_GRADER_SYSTEM_PROMPT
    assert observed_requests[0]["output_config"] == rubric_criterion_output_config()
    assert observed_requests[0]["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert "temperature" not in observed_requests[0]


def test_evaluate_answer_llm_grades_each_criterion_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_requests: list[dict[str, object]] = []
    _patch_anthropic(
        monkeypatch,
        [
            _fake_message(json.dumps({"met": True, "rationale": "present"})),
            _fake_message(json.dumps({"met": False, "rationale": "missing"})),
        ],
        observed_requests=observed_requests,
    )
    config = {
        "answer_field": "rationale",
        "criteria": [
            {"description": "states that hERG is required", "score_delta": 1},
            {"description": "includes a second correct reason", "score_delta": 1},
        ],
    }

    result = asyncio.run(RubricGrader().evaluate_answer_llm({"rationale": "hi"}, config, api_key="k"))

    assert len(observed_requests) == 2
    first_message = observed_requests[0]["messages"][0]
    second_message = observed_requests[1]["messages"][0]
    assert isinstance(first_message, dict)
    assert isinstance(second_message, dict)
    first_prompt = first_message["content"]
    second_prompt = second_message["content"]
    assert isinstance(first_prompt, str)
    assert isinstance(second_prompt, str)
    assert "states that hERG is required" in first_prompt
    assert "includes a second correct reason" not in first_prompt
    assert "states that hERG is required" not in second_prompt
    assert "includes a second correct reason" in second_prompt
    assert result.score == 0.5
    assert result.metrics["judgments"] == [
        {"index": 0, "met": True, "rationale": "present"},
        {"index": 1, "met": False, "rationale": "missing"},
    ]
    assert result.metrics["criterion_parse_attempts"] == {"criterion_0": 1, "criterion_1": 1}


def test_evaluate_answer_llm_retries_until_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    good = json.dumps({"met": True, "rationale": "present"})
    messages = [
        _fake_message(json.dumps({"totally_unexpected": {"nope": 1}})),
        _fake_message("I can't help with that.", stop_reason="refusal"),
        _fake_message(good),
    ]

    _patch_anthropic(monkeypatch, messages)
    config = {
        "answer_field": "rationale",
        "criteria": [{"description": "states that hERG is required", "score_delta": 1}],
    }

    result = asyncio.run(RubricGrader().evaluate_answer_llm({"rationale": "hi"}, config, api_key="k"))

    assert result.passed is True
    assert result.metrics["parse_attempts"] == 3


def test_evaluate_answer_llm_retries_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    good = json.dumps({"met": True, "rationale": "present"})
    monkeypatch.setattr(
        "latch_eval_tools.graders.rubric.TRANSIENT_ANTHROPIC_ERRORS",
        (FakeTransientAnthropicError,),
    )
    _patch_anthropic(monkeypatch, [FakeTransientAnthropicError("rate limited"), _fake_message(good)])
    config = {
        "answer_field": "rationale",
        "criteria": [{"description": "states that hERG is required", "score_delta": 1}],
    }

    result = asyncio.run(RubricGrader().evaluate_answer_llm({"rationale": "hi"}, config, api_key="k"))

    assert result.passed is True
    assert result.metrics["parse_attempts"] == 2


def test_evaluate_answer_llm_rejects_full_rubric_response_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    only_first = json.dumps({"judgments": [{"index": 0, "met": True, "rationale": "present"}]})

    _patch_anthropic(monkeypatch, [_fake_message(only_first)] * 4)
    config = {
        "answer_field": "rationale",
        "criteria": [{"description": "states that hERG is required", "score_delta": 1}],
    }

    with pytest.raises(RubricGraderOutputParseError) as exc_info:
        asyncio.run(RubricGrader().evaluate_answer_llm({"rationale": "hi"}, config, api_key="k"))

    assert "could not parse rubric criterion 0 output" in str(exc_info.value)


def test_anthropic_model_thinking_and_output_config_resolution() -> None:
    assert resolve_anthropic_model_name("anthropic/claude-sonnet-5") == "claude-sonnet-5"
    assert resolve_anthropic_model_name("claude-sonnet-5") == "claude-sonnet-5"
    assert rubric_output_config({"output_config": {"effort": "low"}}) == {
        "format": {
            "type": "json_schema",
            "schema": RubricGraderOutput.model_json_schema(),
        },
        "effort": "low",
    }
    assert rubric_criterion_output_config()["format"]["schema"] == RubricCriterionGraderOutput.model_json_schema()
    assert rubric_output_config({"output_config": {"format": {"type": "text"}, "effort": "low"}}) == {
        "format": {
            "type": "json_schema",
            "schema": RubricGraderOutput.model_json_schema(),
        },
        "effort": "low",
    }
