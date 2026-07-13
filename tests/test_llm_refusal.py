import json
from latch_eval_tools.llm_refusal import detect_llm_refusal


def test_detects_anthropic_usage_policy_refusal_in_trajectory() -> None:
    result = detect_llm_refusal(
        trajectory_data={
            "messages": [
                {"text": "I am unable to respond due to Anthropic usage policy."}
            ]
        }
    )
    assert result is not None
    assert result.provider == "anthropic"
    assert result.source == "trajectory"


def test_detects_openai_content_filter_refusal() -> None:
    result = detect_llm_refusal(
        trajectory_data={"finish_reason": "content_filter"}
    )
    assert result is not None
    assert result.provider == "openai"


def test_returns_none_for_normal_output() -> None:
    assert detect_llm_refusal(trajectory_data={"answer": "42 cells"}) is None


def test_sidecar_events_take_precedence() -> None:
    result = detect_llm_refusal(
        refusal_events_data=[{"provider": "anthropic", "raw_reason": "safety"}]
    )
    assert result is not None
    assert result.source == "refusal_sidecar"
    assert result.provider == "anthropic"


def test_write_refusal_verdict_writes_diagnostic(tmp_path) -> None:
    from latch_eval_tools.harness._cli_runner import (
        REFUSAL_VERDICT_FILENAME,
        _write_refusal_verdict,
    )

    _write_refusal_verdict(
        tmp_path, [{"text": "I am unable to respond due to Anthropic usage policy"}]
    )
    verdict = json.loads((tmp_path / REFUSAL_VERDICT_FILENAME).read_text())
    assert verdict["provider"] == "anthropic"
    assert verdict["source"] == "trajectory"


def test_write_refusal_verdict_writes_null_for_normal_run(tmp_path) -> None:
    from latch_eval_tools.harness._cli_runner import (
        REFUSAL_VERDICT_FILENAME,
        _write_refusal_verdict,
    )

    _write_refusal_verdict(tmp_path, [{"text": "42 cells"}])
    assert json.loads((tmp_path / REFUSAL_VERDICT_FILENAME).read_text()) is None
