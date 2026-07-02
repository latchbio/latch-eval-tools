import json

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
    ]

    for payload in payloads:
        output = parse_rubric_grader_output(payload)

        assert output.judgments[0].index == 0
        assert output.judgments[0].met is True
        assert output.judgments[0].rationale == "present"


def test_rubric_litellm_params_allowlists_configured_thinking() -> None:
    assert rubric_litellm_params({"thinking": {"type": "adaptive"}}) == {
        "allowed_openai_params": ["thinking"]
    }
    assert rubric_litellm_params({}) == {}
