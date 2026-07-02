from copy import deepcopy
import json
from typing import Any

import litellm
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .base import GraderResult, get_nested_value

DEFAULT_TRUNCATION_LENGTH = 512
DEFAULT_MODEL_ID = "anthropic/claude-sonnet-5"
DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "thinking": {
        "type": "adaptive",
        "display": "omitted",
    }
}
RUBRIC_GRADER_SYSTEM_PROMPT = "You are a teacher grading a test response vs a rubric."


def default_model_params() -> dict[str, Any]:
    return deepcopy(DEFAULT_MODEL_PARAMS)


class RubricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    score_delta: float

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("description must be a non-empty string")
        return stripped


class RubricGraderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_field: str = Field(min_length=1)
    criteria: list[RubricCriterion] = Field(min_length=1)
    model_id: str = Field(default=DEFAULT_MODEL_ID, min_length=1)
    model_params: dict[str, Any] = Field(default_factory=default_model_params)
    truncation_length: int = Field(default=DEFAULT_TRUNCATION_LENGTH, gt=0)
    passing_reward_threshold: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("answer_field", "model_id")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("must be a non-empty string")
        return stripped

    @model_validator(mode="after")
    def require_positive_max_score(self) -> "RubricGraderConfig":
        if positive_max_score(self.criteria) <= 0:
            raise ValueError("criteria must include at least one positive score_delta (max_score must be > 0)")
        return self


class RubricCriterionJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    met: bool
    rationale: str = ""


class RubricGraderOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgments: list[RubricCriterionJudgment]


class RubricScoreResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    reward: float
    raw_score: float
    max_score: float
    passed: bool
    field_scores: dict[str, float]


def positive_max_score(criteria: list[RubricCriterion]) -> float:
    return sum(criterion.score_delta for criterion in criteria if criterion.score_delta > 0)


def compute_rubric_reward(config: RubricGraderConfig, output: RubricGraderOutput) -> RubricScoreResult:
    max_score = positive_max_score(config.criteria)
    met_by_index = {judgment.index: judgment.met for judgment in output.judgments}
    raw_score = sum(
        criterion.score_delta
        for index, criterion in enumerate(config.criteria)
        if met_by_index.get(index) is True
    )
    reward = min(max(raw_score / max_score, 0.0), 1.0)
    field_scores = {
        f"criterion_{index}": criterion.score_delta if met_by_index.get(index) is True else 0.0
        for index, criterion in enumerate(config.criteria)
    }
    return RubricScoreResult(
        reward=reward,
        raw_score=raw_score,
        max_score=max_score,
        passed=reward >= config.passing_reward_threshold,
        field_scores=field_scores,
    )


def rubric_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "RubricGraderOutput",
            "schema": RubricGraderOutput.model_json_schema(),
            "strict": True,
        },
    }


def build_rubric_user_prompt(response: str, config: RubricGraderConfig) -> str:
    rubric = "\n".join(
        f"[{index}] {criterion.description}" for index, criterion in enumerate(config.criteria)
    )
    fmt = json.dumps(RubricGraderOutput.model_json_schema(), sort_keys=True)
    return f"""<response>
{response}
</response>

<rubric>
{rubric}
</rubric>

Judge each rubric item independently against the response. Please output your answer in the following json format: {fmt}

Keep each rationale under 256 characters."""


def build_rubric_messages(response: str, config: RubricGraderConfig) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RUBRIC_GRADER_SYSTEM_PROMPT},
        {"role": "user", "content": build_rubric_user_prompt(response, config)},
    ]


def rubric_litellm_params(model_params: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if "thinking" in model_params:
        params["allowed_openai_params"] = ["thinking"]
    return params


def parse_rubric_grader_output(value: Any) -> RubricGraderOutput:
    if isinstance(value, RubricGraderOutput):
        return value
    if isinstance(value, dict):
        return RubricGraderOutput.model_validate(value)
    if isinstance(value, str):
        return RubricGraderOutput.model_validate_json(value)
    raise ValueError(f"expected rubric grader output object or JSON string, got {type(value).__name__}")


class RubricGrader:
    async def evaluate_answer_llm(
        self,
        agent_answer: dict,
        config: dict,
        *,
        api_key: str,
        base_url: str | None = None,
    ) -> GraderResult:
        parsed_config = RubricGraderConfig.model_validate(config)
        answer, found = get_nested_value(agent_answer, parsed_config.answer_field)
        if not found:
            return GraderResult(
                passed=False,
                metrics={"answer_field": parsed_config.answer_field},
                reasoning=f"Rubric: FAIL\n\n  x Agent answer missing required field: {parsed_config.answer_field}",
                agent_answer=agent_answer,
                score=0.0,
                field_scores={},
            )

        response = str(answer)[: parsed_config.truncation_length]
        completion = await litellm.acompletion(
            model=parsed_config.model_id,
            messages=build_rubric_messages(response, parsed_config),
            api_key=api_key,
            base_url=base_url,
            response_format=rubric_response_format(),
            **rubric_litellm_params(parsed_config.model_params),
            **parsed_config.model_params,
        )
        content = completion.choices[0].message.content
        output = parse_rubric_grader_output(content)
        score_result = compute_rubric_reward(parsed_config, output)

        metrics = {
            "answer_field": parsed_config.answer_field,
            "model_id": parsed_config.model_id,
            "model_params": parsed_config.model_params,
            "truncation_length": parsed_config.truncation_length,
            "passing_reward_threshold": parsed_config.passing_reward_threshold,
            "raw_score": score_result.raw_score,
            "max_score": score_result.max_score,
            "reward": score_result.reward,
            "judgments": [judgment.model_dump(mode="json") for judgment in output.judgments],
        }
        return GraderResult(
            passed=score_result.passed,
            metrics=metrics,
            reasoning=format_rubric_reasoning(parsed_config, output, score_result),
            agent_answer=agent_answer,
            score=score_result.reward,
            field_scores=score_result.field_scores,
        )


def format_rubric_reasoning(
    config: RubricGraderConfig,
    output: RubricGraderOutput,
    score_result: RubricScoreResult,
) -> str:
    verdict = "PASS" if score_result.passed else "FAIL"
    lines = [
        f"Rubric: {verdict}",
        f"  reward: {score_result.reward:.4f} (raw_score={score_result.raw_score}, max_score={score_result.max_score})",
    ]
    judgments_by_index = {judgment.index: judgment for judgment in output.judgments}
    for index, criterion in enumerate(config.criteria):
        judgment = judgments_by_index.get(index)
        met = judgment.met if judgment is not None else False
        marker = "+" if met else "x"
        rationale = "" if judgment is None else judgment.rationale
        lines.append(f"  {marker} [{index}] {criterion.description} (delta={criterion.score_delta})")
        if rationale:
            lines.append(f"      rationale: {rationale}")
    return "\n".join(lines)
