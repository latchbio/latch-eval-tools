import json

import pytest

from latch_eval_tools.harness import _cli_runner
from latch_eval_tools.harness.run_summary import build_cli_run_summary

REFUSED_RESULT = {
    "type": "result",
    "subtype": "success",
    "stop_reason": "refusal",
    "result": "",
    "session_id": "session-1",
}
FINISHED_RESULT = {
    "type": "result",
    "subtype": "success",
    "stop_reason": "end_turn",
    "result": "done",
    "session_id": "session-1",
}


def test_read_fallback_api_keys_pops_the_ordered_list() -> None:
    env = {
        "ANTHROPIC_API_KEY": "primary",
        _cli_runner.FALLBACK_API_KEYS_ENV: json.dumps(["second", "third"]),
    }

    assert _cli_runner.read_fallback_api_keys(env) == ["second", "third"]
    assert env == {"ANTHROPIC_API_KEY": "primary"}


def test_read_fallback_api_keys_without_configuration_is_empty() -> None:
    assert _cli_runner.read_fallback_api_keys({}) == []
    assert _cli_runner.read_fallback_api_keys({_cli_runner.FALLBACK_API_KEYS_ENV: ""}) == []


@pytest.mark.parametrize("raw", ['"single-key"', '["", "key"]', "[1]", "{}"])
def test_read_fallback_api_keys_rejects_malformed_values(raw: str) -> None:
    with pytest.raises(ValueError):
        _cli_runner.read_fallback_api_keys({_cli_runner.FALLBACK_API_KEYS_ENV: raw})


def test_refusal_verdict_only_judges_events_after_the_key_switch() -> None:
    trajectory = [REFUSED_RESULT, FINISHED_RESULT]

    recovered = build_cli_run_summary(
        agent_type="claudecode",
        trajectory=trajectory,
        duration_seconds=1.0,
        model_name="anthropic/claude-opus-5",
        refusal_trajectory=trajectory[1:],
    )
    assert recovered.refusal.status == "not_detected"

    unchanged = build_cli_run_summary(
        agent_type="claudecode",
        trajectory=trajectory,
        duration_seconds=1.0,
        model_name="anthropic/claude-opus-5",
    )
    assert unchanged.refusal.status == "detected"
    assert unchanged.refusal.diagnostic is not None
    assert unchanged.refusal.diagnostic.provider == "anthropic"
