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


def test_ignores_refusal_word_inside_paths_within_one_message() -> None:
    # Observed in production: a `find /` dump put
    # "/root/.pi/.refusal_patcher_status.json" and
    # ".../security/policy/README.txt" in the same tool result, so both halves
    # of the marker pair shared one string and a run that merely failed its
    # output contract was reported as a provider refusal.
    result = detect_llm_refusal(
        trajectory_data={
            "messages": [
                {
                    "text": (
                        "/root/.pi/.refusal_patcher_status.json\n"
                        "/root/.pi/agent/auth.json\n"
                        "/etc/java-17-openjdk/security/policy/README.txt\n"
                        "/opt/bismark/license.txt\n"
                    )
                },
                {"text": "Writing eval_answer.json"},
            ]
        }
    )
    assert result is None


def test_ignores_command_output_discussing_refusals() -> None:
    # Observed in production: `git log --oneline` inside a harness checkout
    # listed commit subjects about refusal handling next to a branch mentioning
    # safety, which is prose, not a provider verdict.
    git_log = "\n".join(
        [
            "94de40a fix: stop reading harness refusal artifacts as model refusals",
            "da8207d fix: detect the newer safety-intervention error text as refusals",
            "5cfca90 fix: don't flag recovered fallbacks as LLM refusals",
            "fd8e69d fix: add missing math import in numeric grader",
            "c684f79 Merge pull request #131",
            "e33b105 fix: gate the reward on the policy head",
        ]
    )
    assert detect_llm_refusal(trajectory_data={"messages": [{"text": git_log}]}) is None


def test_detects_refusal_stop_reason_recorded_after_earlier_responses() -> None:
    # Trajectory shape of a refused coding-agent run: one entry per model
    # response, the refused one last, with the API's stop reason under
    # "rawStopReason" while "stopReason" reads "error".
    fallback_notice = (
        "API integrators: you can reduce refusals for your users by configuring "
        "a fallback model — see "
        "https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback"
    )
    result = detect_llm_refusal(
        trajectory_data=[
            {"type": "message_end", "message": {"stopReason": "tool_use"}},
            {
                "type": "tool_result",
                "output": "n rows violating >0.011: 0\nmax deviation 0.0",
            },
            {"type": "message_end", "message": {"stopReason": "pending"}},
            {
                "type": "message_end",
                "message": {
                    "stopReason": "error",
                    "rawStopReason": "refusal",
                    "errorMessage": fallback_notice,
                },
            },
        ]
    )
    assert result is not None
    assert result.provider == "anthropic"
    # Not "pending"/"tool_use" from an earlier response.
    assert result.code == "refusal"
    # Not the agent's own analysis output, which merely says "violating".
    assert result.message == fallback_notice


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
