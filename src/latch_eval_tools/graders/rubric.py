import asyncio
from copy import deepcopy
import random
from typing import Any

from anthropic import APIConnectionError, APIStatusError, APITimeoutError, AsyncAnthropic, RateLimitError
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
RAW_CONTENT_DEBUG_LIMIT = 20000

# Use the native Anthropic API instead of LiteLLM because LiteLLM structured
# output could not be made reliable for this grader; output_config.format gives
# schema-constrained decoding while preserving extended thinking.
GRADER_MAX_RETRIES = 3
GRADER_MAX_ATTEMPTS = GRADER_MAX_RETRIES + 1
RETRY_BASE_DELAY_SECONDS = 0.5
RETRY_MAX_DELAY_SECONDS = 8.0
RETRY_JITTER_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_TOKENS_WITH_THINKING = 16_000
DEFAULT_MAX_TOKENS = 4_096
ANTHROPIC_PROVIDER_PREFIX = "anthropic/"


TRANSIENT_ANTHROPIC_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError)


def retry_backoff_seconds(attempt: int) -> float:
    base = min(RETRY_BASE_DELAY_SECONDS * (2**attempt), RETRY_MAX_DELAY_SECONDS)
    return base + random.uniform(0, RETRY_JITTER_SECONDS)


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


class RubricCriterionGraderOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    met: bool
    rationale: str = ""


class RubricCriterionGradeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    judgment: RubricCriterionJudgment
    attempts_used: int
    finish_reason: Any = None
    had_tool_calls: bool = False


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


def rubric_output_config(model_params: dict[str, Any] | None = None) -> dict[str, Any]:
    return rubric_json_schema_output_config(RubricGraderOutput, model_params)


def rubric_criterion_output_config(model_params: dict[str, Any] | None = None) -> dict[str, Any]:
    return rubric_json_schema_output_config(RubricCriterionGraderOutput, model_params)


def rubric_json_schema_output_config(
    output_model: type[BaseModel],
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_config = {
        "format": {
            "type": "json_schema",
            "schema": output_model.model_json_schema(),
        }
    }
    configured_output_config = (model_params or {}).get("output_config")
    if isinstance(configured_output_config, dict):
        output_config.update(
            {key: value for key, value in configured_output_config.items() if key != "format"}
        )
    return output_config


def build_rubric_user_prompt(response: str, config: RubricGraderConfig) -> str:
    rubric = "\n".join(
        f"[{index}] {criterion.description}" for index, criterion in enumerate(config.criteria)
    )
    return f"""<response>
{response}
</response>

<rubric>
{rubric}
</rubric>

Judge each rubric item independently against the response. Return one judgment per rubric item, using the item's bracketed number as its index.

Keep each rationale under 256 characters."""


def build_rubric_criterion_user_prompt(
    response: str,
    criterion: RubricCriterion,
    criterion_index: int,
) -> str:
    return f"""<response>
{response}
</response>

<criterion index="{criterion_index}">
{criterion.description}
</criterion>

Judge whether this single rubric criterion is met by the response.

Keep the rationale under 256 characters."""


def build_rubric_messages(response: str, config: RubricGraderConfig) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RUBRIC_GRADER_SYSTEM_PROMPT},
        {"role": "user", "content": build_rubric_user_prompt(response, config)},
    ]


def resolve_anthropic_model_name(model_id: str) -> str:
    if model_id.startswith(ANTHROPIC_PROVIDER_PREFIX):
        return model_id[len(ANTHROPIC_PROVIDER_PREFIX) :]
    return model_id


def anthropic_message_text(message: Any) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


async def create_rubric_message(
    client: AsyncAnthropic,
    config: RubricGraderConfig,
    user_prompt: str,
    output_config: dict[str, Any] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": resolve_anthropic_model_name(config.model_id),
        "max_tokens": DEFAULT_MAX_TOKENS_WITH_THINKING if "thinking" in config.model_params else DEFAULT_MAX_TOKENS,
        "system": RUBRIC_GRADER_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
        "output_config": output_config or rubric_output_config(config.model_params),
        "timeout": config.model_params.get("timeout", REQUEST_TIMEOUT_SECONDS),
    }
    if "thinking" in config.model_params:
        kwargs["thinking"] = config.model_params["thinking"]
    return await client.messages.create(**kwargs)


def parse_rubric_grader_output(value: Any) -> RubricGraderOutput:
    if isinstance(value, RubricGraderOutput):
        return value
    if isinstance(value, dict):
        return RubricGraderOutput.model_validate(value)
    if isinstance(value, str):
        return RubricGraderOutput.model_validate_json(value)
    raise ValueError(f"expected rubric grader output object or JSON string, got {type(value).__name__}")


def parse_rubric_criterion_grader_output(value: Any) -> RubricCriterionGraderOutput:
    if isinstance(value, RubricCriterionGraderOutput):
        return value
    if isinstance(value, dict):
        return RubricCriterionGraderOutput.model_validate(value)
    if isinstance(value, str):
        return RubricCriterionGraderOutput.model_validate_json(value)
    raise ValueError(
        f"expected rubric criterion grader output object or JSON string, got {type(value).__name__}"
    )


def validate_judgment_coverage(config: RubricGraderConfig, output: RubricGraderOutput) -> None:
    """Ensure the model judged exactly the rubric criteria, once each.

    Without this, a model that returns judgments for the wrong/missing indices
    would silently score criteria it never judged as unmet, masking a malformed
    grader response as a legitimate low score.
    """
    expected = set(range(len(config.criteria)))
    seen = [judgment.index for judgment in output.judgments]
    seen_set = set(seen)

    if len(seen) != len(seen_set):
        duplicates = sorted({index for index in seen if seen.count(index) > 1})
        raise ValueError(f"rubric grader output has duplicate judgment indices: {duplicates}")

    missing = sorted(expected - seen_set)
    extra = sorted(seen_set - expected)
    if missing or extra:
        raise ValueError(
            f"rubric grader judgment indices {sorted(seen_set)} do not match rubric criteria "
            f"indices {sorted(expected)} (missing={missing}, extra={extra})"
        )


async def grade_rubric_criterion(
    client: AsyncAnthropic,
    config: RubricGraderConfig,
    response: str,
    criterion: RubricCriterion,
    criterion_index: int,
) -> RubricCriterionGradeResult:
    user_prompt = build_rubric_criterion_user_prompt(response, criterion, criterion_index)
    content: Any = None
    finish_reason: Any = None
    had_tool_calls = False
    attempts_used = 0

    for attempt in range(GRADER_MAX_ATTEMPTS):
        attempts_used = attempt + 1
        is_last_attempt = attempt == GRADER_MAX_ATTEMPTS - 1
        try:
            message = await create_rubric_message(
                client,
                config,
                user_prompt,
                rubric_criterion_output_config(config.model_params),
            )
        except TRANSIENT_ANTHROPIC_ERRORS:
            if is_last_attempt:
                raise
            await asyncio.sleep(retry_backoff_seconds(attempt))
            continue
        except APIStatusError as exc:
            if exc.status_code >= 500 and not is_last_attempt:
                await asyncio.sleep(retry_backoff_seconds(attempt))
                continue
            raise

        content = anthropic_message_text(message)
        finish_reason = getattr(message, "stop_reason", None)

        try:
            if finish_reason in {"refusal", "max_tokens"}:
                raise ValueError(f"rubric criterion grader model stopped with stop_reason={finish_reason!r}")
            candidate = parse_rubric_criterion_grader_output(content)
        except (ValidationError, ValueError) as exc:
            if isinstance(exc, RubricGraderOutputParseError):
                parse_error = exc
            else:
                raw_content = content if isinstance(content, str) else repr(content)
                parse_error = RubricGraderOutputParseError(
                    f"could not parse rubric criterion {criterion_index} output after {attempts_used} attempt(s): {exc}\n\n"
                    f"finish_reason={finish_reason!r} had_tool_calls={had_tool_calls}\n"
                    f"raw_content={raw_content[:RAW_CONTENT_DEBUG_LIMIT]}",
                    raw_content=content,
                    finish_reason=finish_reason,
                    had_tool_calls=had_tool_calls,
                )
            if is_last_attempt:
                raise parse_error from exc
            await asyncio.sleep(retry_backoff_seconds(attempt))
            continue

        return RubricCriterionGradeResult(
            judgment=RubricCriterionJudgment(
                index=criterion_index,
                met=candidate.met,
                rationale=candidate.rationale,
            ),
            attempts_used=attempts_used,
            finish_reason=finish_reason,
            had_tool_calls=had_tool_calls,
        )

    raise AssertionError("unreachable: criterion grading loop must return or raise")


class RubricGrader:
    async def evaluate_answer_llm(
        self,
        agent_answer: dict,
        config: dict,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> GraderResult:
        """Grade ``agent_answer`` against a rubric via the Anthropic Messages API.

        When ``api_key`` / ``base_url`` are ``None`` the Anthropic client falls
        back to ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL`` from the
        environment. This lets a caller running inside a managed Anthropic API
        environment (e.g. a Taiga grader container with ``enable_anthropic_api``)
        invoke the grader without threading credentials through.
        """
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
        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        client = AsyncAnthropic(**client_kwargs)

        criterion_results = await asyncio.gather(
            *(
                grade_rubric_criterion(client, parsed_config, response, criterion, criterion_index)
                for criterion_index, criterion in enumerate(parsed_config.criteria)
            )
        )
        output = RubricGraderOutput(
            judgments=[result.judgment for result in sorted(criterion_results, key=lambda result: result.judgment.index)]
        )
        validate_judgment_coverage(parsed_config, output)

        score_result = compute_rubric_reward(parsed_config, output)

        criterion_parse_attempts = {
            f"criterion_{result.judgment.index}": result.attempts_used for result in criterion_results
        }
        finish_reasons = {
            f"criterion_{result.judgment.index}": result.finish_reason for result in criterion_results
        }
        had_tool_calls_by_criterion = {
            f"criterion_{result.judgment.index}": result.had_tool_calls for result in criterion_results
        }
        metrics = {
            "answer_field": parsed_config.answer_field,
            "model_id": parsed_config.model_id,
            "model_params": parsed_config.model_params,
            "truncation_length": parsed_config.truncation_length,
            "passing_reward_threshold": parsed_config.passing_reward_threshold,
            "raw_score": score_result.raw_score,
            "max_score": score_result.max_score,
            "reward": score_result.reward,
            "parse_attempts": sum(criterion_parse_attempts.values()),
            "criterion_parse_attempts": criterion_parse_attempts,
            "finish_reason": finish_reasons,
            "had_tool_calls": any(had_tool_calls_by_criterion.values()),
            "had_tool_calls_by_criterion": had_tool_calls_by_criterion,
            "grading_transport": "anthropic_api",
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
