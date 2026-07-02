import asyncio
import json
import types

import litellm
import pytest
from pydantic import ValidationError

from latch_eval_tools.graders import LLM_GRADER_REGISTRY
from latch_eval_tools.graders.rubric import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_PARAMS,
    DEFAULT_TRUNCATION_LENGTH,
    RUBRIC_GRADER_SYSTEM_PROMPT,
    RubricGrader,
    RubricGraderConfig,
    RubricGraderOutput,
    RubricGraderOutputParseError,
    build_rubric_messages,
    compute_rubric_reward,
    parse_rubric_grader_output,
    rubric_litellm_params,
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
    assert "output your answer in the following json format" in messages[1]["content"]
    assert "Keep each rationale under 256 characters." in messages[1]["content"]


def test_rubric_output_preserves_rationale() -> None:
    output = RubricGraderOutput.model_validate(
        {"judgments": [{"index": 0, "met": True, "rationale": "x" * 300}]}
    )

    assert len(output.judgments[0].rationale) == 300


def test_parse_rubric_grader_output_accepts_structured_output_wrappers() -> None:
    judgment = {"index": 0, "met": True, "rationale": "present"}
    payloads = [
        {"judgments": [judgment]},
        json.dumps({"judgments": [judgment]}),
        json.dumps({"judgments": json.dumps({"judgments": [judgment]})}),
        json.dumps({"judgments": json.dumps([judgment])}),
        json.dumps({"json_value": {"judgments": [judgment]}}),
        json.dumps({"$PARAMETER_NAME": {"judgments": [judgment]}}),
        json.dumps({"$PARAMETER_VALUE": {"judgments": [judgment]}}),
        json.dumps({"arbitrary_provider_wrapper": {"judgments": [judgment]}}),
        json.dumps({"tool_call": {"arguments": json.dumps({"judgments": [judgment]})}}),
        json.dumps([judgment]),
    ]

    for payload in payloads:
        output = parse_rubric_grader_output(payload)

        assert output.judgments[0].index == 0
        assert output.judgments[0].met is True
        assert output.judgments[0].rationale == "present"


def test_parse_rubric_grader_output_accepts_content_block_lists() -> None:
    judgment = {"index": 0, "met": True, "rationale": "present"}
    full_json = json.dumps({"judgments": [judgment]})

    single_block = [{"type": "text", "text": full_json}]
    split_blocks = [
        {"type": "text", "text": '{"judgments":'},
        {"type": "text", "text": json.dumps([judgment]) + "}"},
    ]

    for payload in (single_block, split_blocks):
        output = parse_rubric_grader_output(payload)

        assert output.judgments[0].index == 0
        assert output.judgments[0].met is True
        assert output.judgments[0].rationale == "present"


def test_rubric_output_rejects_empty_judgments() -> None:
    with pytest.raises(ValidationError):
        RubricGraderOutput.model_validate({"judgments": []})

    with pytest.raises(ValidationError):
        parse_rubric_grader_output(json.dumps({"judgments": []}))


def _fake_completion(content: object, *, finish_reason: str = "stop", tool_calls: object = None) -> object:
    message = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    return types.SimpleNamespace(choices=[choice])


def _patch_no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("latch_eval_tools.graders.rubric.asyncio.sleep", fake_sleep)


def _run_rubric_grader(monkeypatch: pytest.MonkeyPatch, completion: object) -> None:
    async def fake_acompletion(**_kwargs: object) -> object:
        return completion

    monkeypatch.setattr(
        "latch_eval_tools.graders.rubric.litellm.acompletion",
        fake_acompletion,
    )
    _patch_no_sleep(monkeypatch)
    config = {
        "answer_field": "rationale",
        "criteria": [{"description": "states that hERG is required", "score_delta": 1}],
    }
    asyncio.run(RubricGrader().evaluate_answer_llm({"rationale": "hi"}, config, api_key="k"))


def test_evaluate_answer_llm_wraps_parse_failure_with_raw_content(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_content = json.dumps({"totally_unexpected": {"nope": 1}})

    with pytest.raises(RubricGraderOutputParseError) as exc_info:
        _run_rubric_grader(monkeypatch, _fake_completion(raw_content, finish_reason="stop"))

    exc = exc_info.value
    assert exc.raw_content == raw_content
    assert exc.finish_reason == "stop"
    assert exc.had_tool_calls is False
    assert "totally_unexpected" in str(exc)


def test_evaluate_answer_llm_reports_none_content(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RubricGraderOutputParseError) as exc_info:
        _run_rubric_grader(
            monkeypatch,
            _fake_completion(None, finish_reason="tool_calls", tool_calls=[{"id": "1"}]),
        )

    exc = exc_info.value
    assert exc.raw_content is None
    assert exc.finish_reason == "tool_calls"
    assert exc.had_tool_calls is True


def test_evaluate_answer_llm_parses_content_block_list(monkeypatch: pytest.MonkeyPatch) -> None:
    judgments = json.dumps({"judgments": [{"index": 0, "met": True, "rationale": "present"}]})
    completion = _fake_completion([{"type": "text", "text": judgments}])

    async def fake_acompletion(**_kwargs: object) -> object:
        return completion

    monkeypatch.setattr(
        "latch_eval_tools.graders.rubric.litellm.acompletion",
        fake_acompletion,
    )
    config = {
        "answer_field": "rationale",
        "criteria": [{"description": "states that hERG is required", "score_delta": 1}],
    }
    result = asyncio.run(RubricGrader().evaluate_answer_llm({"rationale": "hi"}, config, api_key="k"))

    assert result.passed is True
    assert result.metrics["judgments"][0]["met"] is True


def test_parse_rubric_grader_output_strips_code_fences() -> None:
    judgment = {"index": 0, "met": True, "rationale": "present"}
    fenced = "```json\n" + json.dumps({"judgments": [judgment]}) + "\n```"

    output = parse_rubric_grader_output(fenced)

    assert output.judgments[0].index == 0
    assert output.judgments[0].met is True


def test_evaluate_answer_llm_retries_until_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    good = json.dumps({"judgments": [{"index": 0, "met": True, "rationale": "present"}]})
    completions = [
        _fake_completion(json.dumps({"totally_unexpected": {"nope": 1}})),
        _fake_completion(None),
        _fake_completion(good),
    ]

    async def fake_acompletion(**_kwargs: object) -> object:
        return completions.pop(0)

    monkeypatch.setattr("latch_eval_tools.graders.rubric.litellm.acompletion", fake_acompletion)
    _patch_no_sleep(monkeypatch)
    config = {
        "answer_field": "rationale",
        "criteria": [{"description": "states that hERG is required", "score_delta": 1}],
    }

    result = asyncio.run(RubricGrader().evaluate_answer_llm({"rationale": "hi"}, config, api_key="k"))

    assert result.passed is True
    assert result.metrics["parse_attempts"] == 3
    assert completions == []


def test_evaluate_answer_llm_retries_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    good = json.dumps({"judgments": [{"index": 0, "met": True, "rationale": "present"}]})
    calls = {"n": 0}

    async def fake_acompletion(**_kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise litellm.RateLimitError("rate limited", llm_provider="anthropic", model="claude")
        return _fake_completion(good)

    monkeypatch.setattr("latch_eval_tools.graders.rubric.litellm.acompletion", fake_acompletion)
    _patch_no_sleep(monkeypatch)
    config = {
        "answer_field": "rationale",
        "criteria": [{"description": "states that hERG is required", "score_delta": 1}],
    }

    result = asyncio.run(RubricGrader().evaluate_answer_llm({"rationale": "hi"}, config, api_key="k"))

    assert result.passed is True
    assert result.metrics["parse_attempts"] == 2


def test_evaluate_answer_llm_rejects_incomplete_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    only_first = json.dumps({"judgments": [{"index": 0, "met": True, "rationale": "present"}]})

    async def fake_acompletion(**_kwargs: object) -> object:
        return _fake_completion(only_first)

    monkeypatch.setattr("latch_eval_tools.graders.rubric.litellm.acompletion", fake_acompletion)
    _patch_no_sleep(monkeypatch)
    config = {
        "answer_field": "rationale",
        "criteria": [
            {"description": "states that hERG is required", "score_delta": 1},
            {"description": "includes a second correct reason", "score_delta": 1},
        ],
    }

    with pytest.raises(RubricGraderOutputParseError) as exc_info:
        asyncio.run(RubricGrader().evaluate_answer_llm({"rationale": "hi"}, config, api_key="k"))

    assert "missing=[1]" in str(exc_info.value)


def test_rubric_litellm_params_allowlists_configured_thinking() -> None:
    assert rubric_litellm_params({"thinking": {"type": "adaptive"}}) == {
        "allowed_openai_params": ["thinking"]
    }
    assert rubric_litellm_params({}) == {}
