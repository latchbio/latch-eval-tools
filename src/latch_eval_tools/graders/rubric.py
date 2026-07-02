from copy import deepcopy
import json
from typing import Any

import litellm
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

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
MAX_OUTPUT_NORMALIZATION_DEPTH = 8
RAW_CONTENT_DEBUG_LIMIT = 20000


class RubricGraderOutputParseError(ValueError):
    """Raised when the rubric grader model output cannot be parsed.

    Carries the untruncated raw model content and response metadata so a
    downstream failure record (which otherwise only sees a truncated pydantic
    error) contains everything needed to debug a new/unexpected output shape.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_content: Any = None,
        finish_reason: Any = None,
        had_tool_calls: bool = False,
    ) -> None:
        self.raw_content = raw_content
        self.finish_reason = finish_reason
        self.had_tool_calls = had_tool_calls
        super().__init__(message)


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

    judgments: list[RubricCriterionJudgment] = Field(min_length=1)


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


def content_block_text(blocks: list[Any]) -> str | None:
    """Join the text of a provider content-block list (e.g. [{"type": "text", "text": ...}]).

    Returns None when the list does not look like text content blocks, so the
    caller can leave the value untouched rather than fabricating a string.
    """
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            return None
        for key in ("text", "content"):
            candidate = block.get(key)
            if isinstance(candidate, str):
                parts.append(candidate)
                break
    if not parts:
        return None
    return "".join(parts)


def validate_rubric_grader_output_candidate(value: Any) -> RubricGraderOutput | None:
    if isinstance(value, RubricGraderOutput):
        return value
    try:
        if isinstance(value, list):
            return RubricGraderOutput.model_validate({"judgments": value})
        if isinstance(value, dict):
            return RubricGraderOutput.model_validate(value)
    except ValidationError:
        return None
    return None


def normalize_rubric_grader_output(value: Any, *, depth: int = 0) -> Any:
    if depth > MAX_OUTPUT_NORMALIZATION_DEPTH:
        return value

    output = validate_rubric_grader_output_candidate(value)
    if output is not None:
        return output

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if parsed == value:
            return value
        return normalize_rubric_grader_output(parsed, depth=depth + 1)

    if isinstance(value, list):
        text = content_block_text(value)
        if text is not None:
            return normalize_rubric_grader_output(text, depth=depth + 1)
        return value

    if isinstance(value, dict):
        judgments = value.get("judgments")
        if isinstance(judgments, str | dict | list):
            normalized_judgments = normalize_rubric_grader_output(judgments, depth=depth + 1)
            output = validate_rubric_grader_output_candidate(normalized_judgments)
            if output is not None:
                return output
            if isinstance(normalized_judgments, list):
                output = validate_rubric_grader_output_candidate({**value, "judgments": normalized_judgments})
                if output is not None:
                    return output

        candidates: list[RubricGraderOutput] = []
        for nested_value in value.values():
            normalized_value = normalize_rubric_grader_output(nested_value, depth=depth + 1)
            output = validate_rubric_grader_output_candidate(normalized_value)
            if output is not None:
                candidates.append(output)
        if len(candidates) == 1:
            return candidates[0]

    return value


def parse_rubric_grader_output(value: Any) -> RubricGraderOutput:
    normalized = normalize_rubric_grader_output(value)
    if isinstance(normalized, RubricGraderOutput):
        return normalized
    if isinstance(normalized, dict):
        return RubricGraderOutput.model_validate(normalized)
    if isinstance(normalized, list):
        return RubricGraderOutput.model_validate({"judgments": normalized})
    if isinstance(normalized, str):
        return RubricGraderOutput.model_validate_json(normalized)
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
        choice = completion.choices[0]
        message = choice.message
        content = message.content
        try:
            output = parse_rubric_grader_output(content)
        except (ValidationError, ValueError) as exc:
            if isinstance(exc, RubricGraderOutputParseError):
                raise
            raw_content = content if isinstance(content, str) else repr(content)
            raise RubricGraderOutputParseError(
                f"could not parse rubric grader output: {exc}\n\n"
                f"finish_reason={getattr(choice, 'finish_reason', None)!r} "
                f"had_tool_calls={bool(getattr(message, 'tool_calls', None))}\n"
                f"raw_content={raw_content[:RAW_CONTENT_DEBUG_LIMIT]}",
                raw_content=content,
                finish_reason=getattr(choice, "finish_reason", None),
                had_tool_calls=bool(getattr(message, "tool_calls", None)),
            ) from exc
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
