import asyncio
import json
import types

import httpx
import pytest
from anthropic import APIStatusError, APITimeoutError, RateLimitError
from pydantic import ValidationError

from latch_eval_tools.graders import LLM_GRADER_REGISTRY
from latch_eval_tools.graders.rubric import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_PARAMS,
    DEFAULT_TRUNCATION_LENGTH,
    RUBRIC_GRADER_SYSTEM_PROMPT,
    GraderError,
    GraderTransientError,
    RubricCriterionGraderOutput,
    RubricGrader,
    RubricGraderConfig,
    RubricGraderOutput,
    RubricGraderOutputParseError,
    TransientRetryController,
    classify_transient_error,
    compute_rubric_reward,
    resolve_anthropic_model_name,
    retry_after_seconds_from_error,
    rubric_criterion_output_config,
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


def test_llm_registry() -> None:
    assert LLM_GRADER_REGISTRY == {"rubric": RubricGrader}


def test_rubric_output_preserves_rationale() -> None:
    output = RubricGraderOutput.model_validate(
        {"judgments": [{"index": 0, "met": True, "rationale": "x" * 300}]}
    )

    assert len(output.judgments[0].rationale) == 300


def test_rubric_output_rejects_empty_judgments() -> None:
    with pytest.raises(ValidationError):
        RubricGraderOutput.model_validate({"judgments": []})


def test_anthropic_model_and_output_config_resolution() -> None:
    assert resolve_anthropic_model_name("anthropic/claude-sonnet-5") == "claude-sonnet-5"
    assert resolve_anthropic_model_name("claude-sonnet-5") == "claude-sonnet-5"
    assert rubric_criterion_output_config()["format"]["schema"] == RubricCriterionGraderOutput.model_json_schema()
    assert rubric_criterion_output_config({"output_config": {"effort": "low"}}) == {
        "format": {
            "type": "json_schema",
            "schema": RubricCriterionGraderOutput.model_json_schema(),
        },
        "effort": "low",
    }


def _api_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(status_code: int, headers: dict[str, str] | None = None) -> APIStatusError:
    response = httpx.Response(status_code, headers=headers or {}, request=_api_request())
    if status_code == 429:
        return RateLimitError("rate limited", response=response, body=None)
    return APIStatusError(f"status {status_code}", response=response, body=None)


def test_classify_transient_error() -> None:
    info_429 = classify_transient_error(_status_error(429))
    assert info_429 is not None
    assert info_429.status_code == 429
    assert info_429.rate_pressure is True

    info_529 = classify_transient_error(_status_error(529))
    assert info_529 is not None
    assert info_529.status_code == 529
    assert info_529.rate_pressure is True

    info_500 = classify_transient_error(_status_error(500))
    assert info_500 is not None
    assert info_500.status_code == 500
    assert info_500.rate_pressure is False

    info_timeout = classify_transient_error(APITimeoutError(request=_api_request()))
    assert info_timeout is not None
    assert info_timeout.status_code is None
    assert info_timeout.rate_pressure is False

    assert classify_transient_error(_status_error(400)) is None
    assert classify_transient_error(_status_error(401)) is None
    assert classify_transient_error(_status_error(404)) is None
    assert classify_transient_error(_status_error(413)) is None
    assert classify_transient_error(ValueError("nope")) is None


def test_retry_after_seconds_from_error() -> None:
    assert retry_after_seconds_from_error(_status_error(429, {"retry-after": "7"})) == 7.0
    assert retry_after_seconds_from_error(_status_error(429, {"retry-after-ms": "250"})) == 0.25
    assert retry_after_seconds_from_error(_status_error(429, {"retry-after": "nonsense"})) is None
    assert retry_after_seconds_from_error(_status_error(429)) is None


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _controller(budget: float = 30.0) -> tuple[TransientRetryController, _FakeClock]:
    clock = _FakeClock()
    controller = TransientRetryController(budget, time_source=clock.time, sleep=clock.sleep)
    return controller, clock


def _max_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "latch_eval_tools.graders.rubric.random",
        types.SimpleNamespace(uniform=lambda _low, high: high),
    )


def test_controller_budget_exhaustion_raises_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    _max_jitter(monkeypatch)
    controller, clock = _controller(budget=30.0)
    error = _status_error(529)
    info = classify_transient_error(error)
    assert info is not None

    async def run() -> None:
        # Full-jitter caps: 2, 4, 8, 16 -> exactly the 30s budget.
        for attempt in range(4):
            await controller.backoff(attempt, info, error)
        with pytest.raises(GraderTransientError) as exc_info:
            await controller.backoff(4, info, error)
        assert exc_info.value.status_code == 529
        assert exc_info.value.sleep_used_seconds == pytest.approx(30.0)

    asyncio.run(run())
    assert clock.now == pytest.approx(30.0)


def test_controller_serializes_on_rate_pressure(monkeypatch: pytest.MonkeyPatch) -> None:
    _max_jitter(monkeypatch)
    controller, _clock = _controller()
    error = _status_error(429)
    info = classify_transient_error(error)
    assert info is not None
    assert controller.serialized is False

    asyncio.run(controller.backoff(0, info, error))

    assert controller.serialized is True


def test_controller_overlapping_backoffs_share_one_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    _max_jitter(monkeypatch)
    controller, clock = _controller(budget=30.0)
    error = _status_error(529)
    info = classify_transient_error(error)
    assert info is not None

    async def run() -> None:
        # Two tasks failing at the same instant with the same delay charge the
        # budget once: the second backoff's window is already covered.
        await controller.backoff(0, info, error)
        assert controller.sleep_used_seconds == pytest.approx(2.0)
        clock.now = 0.0
        await controller.backoff(0, info, error)

    asyncio.run(run())
    assert controller.sleep_used_seconds == pytest.approx(2.0)
    assert clock.now == pytest.approx(2.0)


def test_controller_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    _max_jitter(monkeypatch)
    controller, clock = _controller(budget=30.0)
    error = _status_error(429, {"retry-after": "11"})
    info = classify_transient_error(error)
    assert info is not None
    assert info.retry_after_seconds == 11.0

    # retry-after (11s) exceeds the attempt-0 jitter cap (2s) and wins.
    asyncio.run(controller.backoff(0, info, error))

    assert controller.sleep_used_seconds == pytest.approx(11.0)
    assert clock.now == pytest.approx(11.0)


def _fake_message(content: str, *, stop_reason: str = "end_turn") -> object:
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text=content)],
        stop_reason=stop_reason,
    )


def _patch_no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("latch_eval_tools.graders.rubric.asyncio.sleep", fake_sleep)


def _patch_fake_clock_controller(monkeypatch: pytest.MonkeyPatch, budget: float = 30.0) -> _FakeClock:
    clock = _FakeClock()

    def factory(sleep_budget_seconds: float = budget) -> TransientRetryController:
        return TransientRetryController(
            sleep_budget_seconds,
            time_source=clock.time,
            sleep=clock.sleep,
        )

    monkeypatch.setattr("latch_eval_tools.graders.rubric.TransientRetryController", factory)
    return clock


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
    assert isinstance(exc, GraderError)


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
    assert observed_clients == [{"max_retries": 0, "api_key": "k"}]
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


def test_evaluate_answer_llm_recovers_from_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    good = json.dumps({"met": True, "rationale": "present"})
    observed_requests: list[dict[str, object]] = []
    _patch_anthropic(
        monkeypatch,
        [_status_error(529), _status_error(429), _fake_message(good)],
        observed_requests=observed_requests,
    )
    _patch_fake_clock_controller(monkeypatch)
    _max_jitter(monkeypatch)
    config = {
        "answer_field": "rationale",
        "criteria": [{"description": "states that hERG is required", "score_delta": 1}],
    }

    result = asyncio.run(RubricGrader().evaluate_answer_llm({"rationale": "hi"}, config, api_key="k"))

    assert result.passed is True
    assert len(observed_requests) == 3


def test_evaluate_answer_llm_raises_transient_after_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_anthropic(monkeypatch, [_status_error(529) for _ in range(20)])
    clock = _patch_fake_clock_controller(monkeypatch, budget=30.0)
    _max_jitter(monkeypatch)
    config = {
        "answer_field": "rationale",
        "criteria": [{"description": "states that hERG is required", "score_delta": 1}],
    }

    with pytest.raises(GraderTransientError) as exc_info:
        asyncio.run(RubricGrader().evaluate_answer_llm({"rationale": "hi"}, config, api_key="k"))

    assert exc_info.value.status_code == 529
    assert exc_info.value.sleep_used_seconds == pytest.approx(30.0)
    assert clock.now == pytest.approx(30.0)


def test_evaluate_answer_llm_raises_grader_error_on_bad_request(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_requests: list[dict[str, object]] = []
    _patch_anthropic(monkeypatch, [_status_error(400)], observed_requests=observed_requests)
    config = {
        "answer_field": "rationale",
        "criteria": [{"description": "states that hERG is required", "score_delta": 1}],
    }

    with pytest.raises(GraderError) as exc_info:
        asyncio.run(RubricGrader().evaluate_answer_llm({"rationale": "hi"}, config, api_key="k"))

    assert not isinstance(exc_info.value, GraderTransientError)
    assert "non-retryable" in str(exc_info.value)
    assert len(observed_requests) == 1


def test_evaluate_answer_llm_prioritizes_transient_over_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    # One criterion hits a 400 (fatal), the other exhausts the budget on 529s.
    # The transient error must win so callers schedule a retry instead of
    # permanently recording a system error.
    messages: list[object] = [_status_error(400)]
    messages.extend(_status_error(529) for _ in range(20))
    _patch_anthropic(monkeypatch, messages)
    _patch_fake_clock_controller(monkeypatch, budget=30.0)
    _max_jitter(monkeypatch)
    config = {
        "answer_field": "rationale",
        "criteria": [
            {"description": "states that hERG is required", "score_delta": 1},
            {"description": "includes a second correct reason", "score_delta": 1},
        ],
    }

    with pytest.raises(GraderTransientError):
        asyncio.run(RubricGrader().evaluate_answer_llm({"rationale": "hi"}, config, api_key="k"))
