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


CLAUDE_CODE_SAFEGUARD_BANNER = (
    "API Error: Opus 5 (1M context)'s safeguards flagged this message "
    "(https://www.anthropic.com/legal/aup). Our intentionally broad safeguards "
    "allow us to deliver more capabilities faster, but can sometimes flag "
    "legitimate coding, cybersecurity, and biology tasks. Claude Code can't "
    "respond to this message with Opus 5 (1M context).\n\n"
    "Double press esc to edit your last message, or try a different model with "
    "/model.\n\n"
    "Send feedback with /feedback or learn more: "
    "https://support.claude.com/en/articles/16049681\n\n"
    "Details: `[bio]`\n\n"
    "Request ID: req_011CeRsSEdJjzppKDYQdAFkg"
)


def test_detects_claude_code_safeguard_banner_in_trajectory() -> None:
    # The claude-code banner names neither "usage policy" nor "refusal", so it
    # used to read as a plain harness crash and the rollout was retried.
    result = detect_llm_refusal(
        trajectory_data=[
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "api_error_status": None,
                "result": CLAUDE_CODE_SAFEGUARD_BANNER,
            }
        ]
    )
    assert result is not None
    assert result.provider == "anthropic"
    assert result.source == "trajectory"
    assert result.code == "bio"
    assert "safeguards flagged this message" in result.message


def test_detects_claude_code_safeguard_banner_in_agent_error() -> None:
    result = detect_llm_refusal(agent_error=CLAUDE_CODE_SAFEGUARD_BANNER)
    assert result is not None
    assert result.provider == "anthropic"
    assert result.source == "agent_output"
    assert result.code == "bio"


def test_claude_code_safeguard_banner_without_category_still_detected() -> None:
    result = detect_llm_refusal(
        agent_error=(
            "API Error: Opus 5's safeguards flagged this message "
            "(https://www.anthropic.com/legal/aup)."
        )
    )
    assert result is not None
    assert result.provider == "anthropic"
    assert result.code == "refusal"


def test_detects_truncated_claude_code_safeguard_banner_in_workflow_error() -> None:
    # The workflow error surfaces a log tail that is cut mid-URL, so detection
    # cannot depend on the full AUP link or the trailing can't-respond sentence.
    result = detect_llm_refusal(
        workflow_error_data=(
            "Error running v2-code-agent run rollout_1: agent harness failed: "
            "RuntimeError: claudecode exited with code 1; log_tail="
            '"subtype":"success","api_error_status":null,"result":"API Error: '
            "Opus 5's safeguards flagged this message "
            "(https://www.anthropic.com/legal...[truncated]"
        )
    )
    assert result is not None
    assert result.provider == "anthropic"
    assert result.source == "workflow_error"


def test_ignores_safeguard_marker_split_across_unrelated_messages() -> None:
    # "safeguards flagged this message" alone is not enough: it has to land in
    # the same string as the banner's URL or its can't-respond sentence.
    result = detect_llm_refusal(
        trajectory_data={
            "messages": [
                {"text": "The paper explains how safeguards flagged this message."},
                {"text": "See https://www.anthropic.com/legal/aup for details."},
            ]
        }
    )
    assert result is None


def test_detects_openai_content_filter_refusal() -> None:
    result = detect_llm_refusal(trajectory_data={"finish_reason": "content_filter"})
    assert result is not None
    assert result.provider == "openai"


def test_returns_none_for_normal_output() -> None:
    assert detect_llm_refusal(trajectory_data={"answer": "42 cells"}) is None


def test_ignores_refusal_markers_split_across_unrelated_messages() -> None:
    # Observed in production: a bash `ls` dump listed
    # ".refusal_patcher_status.json" while an unrelated turn mentioned a
    # "policy", so a run that merely failed its output contract was reported as
    # a provider refusal (and therefore never retried).
    result = detect_llm_refusal(
        trajectory_data={
            "messages": [
                {"text": "/root/.pi/.refusal_patcher_status.json\t4 KB"},
                {"text": "Applying the sample QC policy to the remaining pixels."},
                {"text": "Writing eval_answer.json"},
            ]
        }
    )
    assert result is None


def test_detects_refusal_when_markers_share_one_message() -> None:
    result = detect_llm_refusal(
        trajectory_data={
            "messages": [
                {"text": "/root/.pi/.refusal_patcher_status.json\t4 KB"},
                {"text": "Refusal: this request violates our safety policy."},
            ]
        }
    )
    assert result is not None
    assert result.provider == "unknown"
    assert result.message == "Refusal: this request violates our safety policy."


def test_sidecar_events_take_precedence() -> None:
    result = detect_llm_refusal(
        refusal_events_data=[{"provider": "anthropic", "raw_reason": "safety"}]
    )
    assert result is not None
    assert result.source == "refusal_sidecar"
    assert result.provider == "anthropic"


def test_diagnostic_bounds_large_trajectory_fields_for_agent_completion() -> None:
    hostile_text = ("\x01" * 20_000) + ('"\\🧪' * 20_000)
    result = detect_llm_refusal(
        trajectory_data={
            "code": hostile_text,
            "message": (
                f"I am unable to respond due to Anthropic usage policy. {hostile_text}"
            ),
        }
    )

    assert result is not None
    assert result.code is not None
    assert result.raw_excerpt is not None
    assert result.code.endswith("…")
    assert result.message.endswith("…")
    assert result.raw_excerpt.endswith("…")
    assert len(json.dumps(result.code, ensure_ascii=False).encode("utf-8")) <= 512
    assert len(json.dumps(result.message, ensure_ascii=False).encode("utf-8")) <= 4096
    assert (
        len(json.dumps(result.raw_excerpt, ensure_ascii=False).encode("utf-8")) <= 4096
    )

    completion_details = {
        "run_summary": {
            "schema_version": 1,
            "metrics": {
                "duration_seconds": 1.0,
                "turn_count": 1,
                "step_count": 1,
                "usage": {},
                "total_cost_usd": None,
                "cost_source": None,
                "pricing_version": None,
            },
            "refusal": {
                "status": "detected",
                "diagnostic": result.model_dump(mode="json"),
            },
        }
    }
    assert (
        len(json.dumps(completion_details, ensure_ascii=False).encode("utf-8"))
        <= 32_768
    )


def test_diagnostic_bounds_large_sidecar_and_agent_error_fields() -> None:
    hostile_text = '\x02"\\🧪' * 20_000
    sidecar_result = detect_llm_refusal(
        refusal_events_data=[
            {
                "provider": "anthropic",
                "raw_reason": hostile_text,
                "explanation": (
                    "I am unable to respond due to Anthropic usage policy. "
                    f"{hostile_text}"
                ),
            }
        ]
    )
    error_result = detect_llm_refusal(
        agent_error=(
            '{"code":"invalid_prompt","message":'
            f'"OpenAI limited access for safety reasons {hostile_text}"'
            "}"
        )
    )

    assert sidecar_result is not None
    assert sidecar_result.code is not None
    assert sidecar_result.raw_excerpt is not None
    assert sidecar_result.code.endswith("…")
    assert sidecar_result.message.endswith("…")
    assert error_result is not None
    assert error_result.message.endswith("…")
    assert error_result.raw_excerpt is not None

    serialized = json.dumps(
        {
            "sidecar": sidecar_result.model_dump(mode="json"),
            "agent_error": error_result.model_dump(mode="json"),
        },
        ensure_ascii=False,
    )
    assert len(serialized.encode("utf-8")) <= 32_768


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
