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
