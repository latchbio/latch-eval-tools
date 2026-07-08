import asyncio
from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Any, Optional

from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic, RateLimitError
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
#
# Retry policy has two independent paths:
# - Parse/refusal errors: fast fixed-attempt retries (the model produced a bad
#   output; waiting longer does not help).
# - Transient API errors (429/529/5xx/timeouts/connection): full-jitter backoff
#   drawn from a budget of cooldown wall-clock time shared across all criterion
#   tasks of one rubric. The SDK's own retries are disabled (max_retries=0) so
#   this policy is the only retry layer. On budget exhaustion the grader raises
#   GraderTransientError so callers can schedule a much longer retry.
GRADER_MAX_RETRIES = 3
GRADER_MAX_ATTEMPTS = GRADER_MAX_RETRIES + 1
RETRY_BASE_DELAY_SECONDS = 0.5
RETRY_MAX_DELAY_SECONDS = 8.0
RETRY_JITTER_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 40.0
TRANSIENT_SLEEP_BUDGET_SECONDS = 30.0
TRANSIENT_BACKOFF_BASE_SECONDS = 2.0
TRANSIENT_BACKOFF_CAP_SECONDS = 20.0
DEFAULT_MAX_TOKENS_WITH_THINKING = 16_000
DEFAULT_MAX_TOKENS = 4_096
ANTHROPIC_PROVIDER_PREFIX = "anthropic/"


class GraderError(Exception):
    """Non-retryable grader infrastructure failure.

    Raised for failures where retrying will not help: bad requests (4xx other
    than rate limits), auth/config problems, and model outputs that repeatedly
    fail to parse. Callers should record a system error, not a score.
    """


class GraderTransientError(Exception):
    """Retryable grader infrastructure failure (rate limit / overload / 5xx / timeout).

    Raised once the shared in-process backoff budget is exhausted. Callers are
    expected to retry the whole grade much later (minutes, not seconds).
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        sleep_used_seconds: float = 0.0,
    ) -> None:
        self.status_code = status_code
        self.sleep_used_seconds = sleep_used_seconds
        super().__init__(message)


@dataclass(frozen=True)
class TransientErrorInfo:
    status_code: Optional[int]
    retry_after_seconds: Optional[float]
    rate_pressure: bool


def retry_after_seconds_from_error(exc: Exception) -> Optional[float]:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            return float(retry_after_ms) / 1000.0
        except ValueError:
            pass
    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return None


def classify_transient_error(exc: Exception) -> Optional[TransientErrorInfo]:
    """Classify an Anthropic SDK error as transient (retryable) or not.

    Returns ``None`` for non-retryable errors (bad request, auth, etc.).
    ``rate_pressure`` marks 429/529, which additionally serialize the rubric's
    remaining criterion requests.
    """
    if isinstance(exc, RateLimitError):
        return TransientErrorInfo(
            status_code=429,
            retry_after_seconds=retry_after_seconds_from_error(exc),
            rate_pressure=True,
        )
    if isinstance(exc, APIStatusError):
        if exc.status_code >= 500 or exc.status_code in (408, 409):
            return TransientErrorInfo(
                status_code=exc.status_code,
                retry_after_seconds=retry_after_seconds_from_error(exc),
                rate_pressure=exc.status_code == 529,
            )
        return None
    if isinstance(exc, APIConnectionError):
        # Includes APITimeoutError (request timeouts) and connection failures.
        return TransientErrorInfo(status_code=None, retry_after_seconds=None, rate_pressure=False)
    return None


class TransientRetryController:
    """Coordinates transient-error backoff across the criterion tasks of one rubric.

    All criterion tasks share one cooldown window and one wall-clock sleep
    budget: overlapping failures extend the same cooldown instead of each
    charging the budget, and every task waits out the active cooldown before
    sending its next request. After the first 429/529, requests additionally
    run one at a time.
    """

    def __init__(
        self,
        sleep_budget_seconds: float = TRANSIENT_SLEEP_BUDGET_SECONDS,
        *,
        time_source: Any = None,
        sleep: Any = None,
    ) -> None:
        self.sleep_budget_seconds = sleep_budget_seconds
        self.sleep_used_seconds = 0.0
        self.serialized = False
        self._serial_lock = asyncio.Lock()
        self._cooldown_until = 0.0
        # Injectable for deterministic tests; production uses the event loop clock.
        self._time = time_source if time_source is not None else (lambda: asyncio.get_running_loop().time())
        self._sleep = sleep if sleep is not None else asyncio.sleep

    async def _wait_for_cooldown(self) -> None:
        while True:
            now = self._time()
            if now >= self._cooldown_until:
                return
            await self._sleep(self._cooldown_until - now)

    async def send(self, request: Any) -> Any:
        if self.serialized:
            async with self._serial_lock:
                await self._wait_for_cooldown()
                return await request()
        await self._wait_for_cooldown()
        return await request()

    async def backoff(self, attempt: int, info: TransientErrorInfo, cause: Exception) -> None:
        """Extend the shared cooldown for a transient error and wait it out.

        Raises :class:`GraderTransientError` when extending the cooldown would
        exceed the shared sleep budget.
        """
        if info.rate_pressure:
            self.serialized = True

        delay = random.uniform(
            0.0,
            min(TRANSIENT_BACKOFF_CAP_SECONDS, TRANSIENT_BACKOFF_BASE_SECONDS * (2**attempt)),
        )
        if info.retry_after_seconds is not None and info.retry_after_seconds > delay:
            delay = info.retry_after_seconds

        now = self._time()
        cooldown_start = max(self._cooldown_until, now)
        extension = (now + delay) - cooldown_start
        if extension > 0:
            remaining = self.sleep_budget_seconds - self.sleep_used_seconds
            if remaining <= 0:
                raise GraderTransientError(
                    f"rubric grader exhausted its {self.sleep_budget_seconds:.0f}s transient backoff budget "
                    f"(last error: {cause})",
                    status_code=info.status_code,
                    sleep_used_seconds=self.sleep_used_seconds,
                ) from cause
            self.sleep_used_seconds += min(extension, remaining)
            self._cooldown_until = cooldown_start + min(extension, remaining)
        await self._wait_for_cooldown()


def retry_backoff_seconds(attempt: int) -> float:
    base = min(RETRY_BASE_DELAY_SECONDS * (2**attempt), RETRY_MAX_DELAY_SECONDS)
    return base + random.uniform(0, RETRY_JITTER_SECONDS)


class RubricGraderOutputParseError(GraderError, ValueError):
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
        "output_config": output_config or rubric_criterion_output_config(config.model_params),
        "timeout": config.model_params.get("timeout", REQUEST_TIMEOUT_SECONDS),
    }
    if "thinking" in config.model_params:
        kwargs["thinking"] = config.model_params["thinking"]
    return await client.messages.create(**kwargs)


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
    controller: TransientRetryController,
) -> RubricCriterionGradeResult:
    user_prompt = build_rubric_criterion_user_prompt(response, criterion, criterion_index)
    output_config = rubric_criterion_output_config(config.model_params)
    had_tool_calls = False
    attempts_used = 0
    parse_failures = 0
    transient_attempt = 0

    while True:
        attempts_used += 1
        try:
            message = await controller.send(
                lambda: create_rubric_message(client, config, user_prompt, output_config)
            )
        except (APIStatusError, APIConnectionError) as exc:
            info = classify_transient_error(exc)
            if info is None:
                raise GraderError(
                    f"rubric criterion {criterion_index} grading failed with non-retryable API error: {exc}"
                ) from exc
            await controller.backoff(transient_attempt, info, exc)
            transient_attempt += 1
            continue

        content = anthropic_message_text(message)
        finish_reason = getattr(message, "stop_reason", None)

        try:
            if finish_reason in {"refusal", "max_tokens"}:
                raise ValueError(f"rubric criterion grader model stopped with stop_reason={finish_reason!r}")
            candidate = parse_rubric_criterion_grader_output(content)
        except (ValidationError, ValueError) as exc:
            parse_failures += 1
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
            if parse_failures >= GRADER_MAX_ATTEMPTS:
                raise parse_error from exc
            await asyncio.sleep(retry_backoff_seconds(parse_failures - 1))
            continue

        return RubricCriterionGradeResult(
            judgment=RubricCriterionJudgment(
                index=criterion_index,
                met=candidate.met,
                rationale=candidate.rationale,
            ),
            attempts_used=attempts_used,
            finish_reason=finish_reason,
        )


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
        # max_retries=0: the TransientRetryController owns the whole retry
        # policy; the SDK's hidden inner retries would multiply attempts and
        # defeat the shared backoff budget.
        client = AsyncAnthropic(max_retries=0, **client_kwargs)

        controller = TransientRetryController()
        settled = await asyncio.gather(
            *(
                grade_rubric_criterion(client, parsed_config, response, criterion, criterion_index, controller)
                for criterion_index, criterion in enumerate(parsed_config.criteria)
            ),
            return_exceptions=True,
        )
        errors = [item for item in settled if isinstance(item, BaseException)]
        if errors:
            for error in errors:
                if isinstance(error, GraderTransientError):
                    raise error
            for error in errors:
                if isinstance(error, GraderError):
                    raise error
            raise errors[0]
        criterion_results = [item for item in settled if isinstance(item, RubricCriterionGradeResult)]
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
