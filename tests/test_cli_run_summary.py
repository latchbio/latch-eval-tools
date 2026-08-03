import json

import pytest

from latch_eval_tools.harness._cli_runner import (
    _extract_metadata,
    _read_codex_sidecar_events,
    _read_pi_refusal_events,
)
from latch_eval_tools.harness.run_summary import (
    HARNESS_PRICING_VERSION,
    build_cli_run_summary,
    build_miniswe_run_summary,
)


def test_claude_summary_uses_result_event_and_counts_tool_calls() -> None:
    trajectory = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Working"},
                    {"type": "tool_use", "id": "tool-1", "name": "Bash"},
                    {"type": "tool_use", "id": "tool-2", "name": "Read"},
                ],
            },
        },
        {
            "type": "result",
            "session_id": "claude-session",
            "num_turns": 3,
            "total_cost_usd": 1.25,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 40,
                "cache_read_input_tokens": 60,
                "cache_creation_input_tokens": 10,
            },
        },
    ]

    metadata = _extract_metadata(
        "claudecode",
        trajectory,
        12.345,
        "anthropic/claude-opus-4-8",
        False,
        600,
        None,
        False,
        0,
        1024,
    )

    assert metadata["total_cost"] == 1.25
    assert metadata["n_turns"] == 3
    assert metadata["n_steps"] == 2
    assert metadata["usage"] == trajectory[-1]["usage"]
    assert metadata["run_summary"] == {
        "schema_version": 1,
        "metrics": {
            "duration_seconds": 12.345,
            "turn_count": 3,
            "step_count": 2,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 40,
                "cache_read_tokens": 60,
                "cache_write_tokens": 10,
                "reasoning_tokens": None,
            },
            "total_cost_usd": 1.25,
            "cost_source": "provider_reported",
            "pricing_version": None,
        },
        "refusal": {
            "status": "not_detected",
            "diagnostic": None,
        },
    }


def test_codex_summary_reads_authoritative_local_sidecar(tmp_path) -> None:
    thread_id = "019cafe0-1111-7222-8333-123456789abc"
    trajectory = [
        {"type": "thread.started", "thread_id": thread_id},
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 7, "output_tokens": 3},
        },
    ]
    (tmp_path / "trajectory.json").write_text(json.dumps(trajectory))
    codex_dir = tmp_path / ".codex" / "sessions"
    codex_dir.mkdir(parents=True)
    (codex_dir / f"rollout-2026-07-29T00-00-00-{thread_id}.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "large output"}
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "call_id": "call-1",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "call_id": "call-2",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 1000,
                                    "cached_input_tokens": 400,
                                    "output_tokens": 200,
                                    "reasoning_output_tokens": 75,
                                }
                            },
                        },
                    }
                ),
            ]
        )
    )

    sidecar_events = _read_codex_sidecar_events(tmp_path, trajectory)
    assert sidecar_events is not None
    assert len(sidecar_events) == 3

    metadata = _extract_metadata(
        "openaicodex",
        trajectory,
        4.5,
        "openai/gpt-5.4",
        False,
        600,
        None,
        False,
        0,
        1024,
        codex_sidecar_events=sidecar_events,
    )
    metrics = metadata["run_summary"]["metrics"]

    assert metrics["turn_count"] == 1
    assert metrics["step_count"] == 2
    assert metrics["usage"] == {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_read_tokens": 400,
        "cache_write_tokens": None,
        "reasoning_tokens": 75,
    }
    assert metrics["total_cost_usd"] == pytest.approx(0.0046)
    assert metrics["cost_source"] == "latch_eval_tools_pricing"
    assert metrics["pricing_version"] == HARNESS_PRICING_VERSION
    assert metadata["n_steps"] == 2
    assert metadata["usage"] == {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_read_tokens": 400,
        "reasoning_tokens": 75,
    }


def test_codex_stream_fallback_preserves_cached_input_usage() -> None:
    summary = build_cli_run_summary(
        agent_type="openaicodex",
        trajectory=[
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 1000,
                    "cached_input_tokens": 400,
                    "output_tokens": 200,
                },
            }
        ],
        duration_seconds=4.5,
        model_name="openai/gpt-5.4",
    )

    assert summary.metrics.usage.cache_read_tokens == 400
    assert summary.metrics.total_cost_usd == pytest.approx(0.0046)


def test_pi_summary_combines_trajectory_metrics_and_refusal_sidecar(
    tmp_path,
) -> None:
    trajectory = [
        {"type": "session", "id": "pi-session"},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "tool-1",
                        "name": "bash",
                    },
                    {
                        "type": "toolCall",
                        "id": "tool-2",
                        "name": "read",
                    },
                ],
                "usage": {
                    "input": 500,
                    "output": 100,
                    "cacheRead": 200,
                    "cacheWrite": 50,
                    "reasoning": 25,
                    "cost": {"total": 0.0125},
                },
            },
        },
        {"type": "turn_end"},
    ]
    pi_dir = tmp_path / ".pi"
    pi_dir.mkdir()
    (pi_dir / "refusal_events.jsonl").write_text(
        json.dumps(
            {
                "provider": "google",
                "raw_reason": "SAFETY",
                "explanation": "Prompt was blocked by the safety policy.",
            }
        )
    )
    refusal_events = _read_pi_refusal_events(tmp_path)
    assert refusal_events is not None

    metadata = _extract_metadata(
        "pi",
        trajectory,
        8.0,
        "google/gemini-3.1-pro",
        False,
        600,
        None,
        False,
        0,
        1024,
        refusal_events=refusal_events,
    )
    run_summary = metadata["run_summary"]

    assert run_summary["metrics"] == {
        "duration_seconds": 8.0,
        "turn_count": 1,
        "step_count": 2,
        "usage": {
            "input_tokens": 500,
            "output_tokens": 100,
            "cache_read_tokens": 200,
            "cache_write_tokens": 50,
            "reasoning_tokens": 25,
        },
        "total_cost_usd": 0.0125,
        "cost_source": "provider_reported",
        "pricing_version": None,
    }
    assert run_summary["refusal"]["status"] == "detected"
    assert run_summary["refusal"]["diagnostic"] == {
        "kind": "llm_refusal",
        "provider": "google",
        "code": "SAFETY",
        "message": "Prompt was blocked by the safety policy.",
        "source": "refusal_sidecar",
        "raw_excerpt": json.dumps(refusal_events[0])[:1000],
    }
    assert metadata["n_steps"] == 2
    assert metadata["total_cost"] == 0.0125


def test_empty_run_has_not_evaluated_refusal_and_unknown_counts() -> None:
    run_summary = build_cli_run_summary(
        agent_type="claudecode",
        trajectory=[],
        duration_seconds=0,
        model_name=None,
    )

    assert run_summary.metrics.turn_count is None
    assert run_summary.metrics.step_count is None
    assert run_summary.refusal.status == "not_evaluated"
    assert run_summary.refusal.diagnostic is None


def test_pi_calculates_cost_when_harness_does_not_report_one() -> None:
    run_summary = build_cli_run_summary(
        agent_type="pi",
        trajectory=[
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "usage": {
                        "input": 1000,
                        "output": 200,
                        "cacheRead": 400,
                        "cacheWrite": 100,
                    },
                },
            }
        ],
        duration_seconds=1,
        model_name="anthropic/claude-opus-4-8",
    )

    assert run_summary.metrics.total_cost_usd == pytest.approx(0.010825)
    assert run_summary.metrics.cost_source == "latch_eval_tools_pricing"
    assert run_summary.metrics.pricing_version == HARNESS_PRICING_VERSION


def test_miniswe_summary_uses_serialized_messages_and_agent_counters() -> None:
    run_summary = build_miniswe_run_summary(
        serialized_trajectory={
            "trajectory_format": "mini-swe-agent-1.1",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": "first turn",
                    "extra": {
                        "response": {
                            "usage": {
                                "prompt_tokens": 100,
                                "completion_tokens": 20,
                                "prompt_tokens_details": {
                                    "cached_tokens": 40,
                                },
                                "completion_tokens_details": {
                                    "reasoning_tokens": 5,
                                },
                            }
                        }
                    },
                },
                {"role": "tool", "content": "observation"},
                {
                    "object": "response",
                    "output": [{"type": "message", "content": []}],
                    "usage": {
                        "input_tokens": 200,
                        "output_tokens": 50,
                        "input_tokens_details": {
                            "cached_tokens": 60,
                        },
                        "output_tokens_details": {
                            "reasoning_tokens": 7,
                        },
                        "cache_creation_input_tokens": 30,
                    },
                    "extra": {
                        "cost": 0.1,
                    },
                },
            ],
        },
        duration_seconds=9.5,
        total_cost_usd=0.25,
        step_count=2,
    )

    assert run_summary.model_dump(mode="json") == {
        "schema_version": 1,
        "metrics": {
            "duration_seconds": 9.5,
            "turn_count": 2,
            "step_count": 2,
            "usage": {
                "input_tokens": 300,
                "output_tokens": 70,
                "cache_read_tokens": 100,
                "cache_write_tokens": 30,
                "reasoning_tokens": 12,
            },
            "total_cost_usd": 0.25,
            "cost_source": "provider_reported",
            "pricing_version": None,
        },
        "refusal": {
            "status": "not_detected",
            "diagnostic": None,
        },
    }


def test_miniswe_summary_detects_refusal_from_structured_agent_error() -> None:
    run_summary = build_miniswe_run_summary(
        serialized_trajectory={"messages": []},
        duration_seconds=2,
        total_cost_usd=0,
        step_count=0,
        agent_error=(
            "OpenAI invalid_prompt: limited access to this content for safety reasons"
        ),
    )

    assert run_summary.refusal.status == "detected"
    assert run_summary.refusal.diagnostic is not None
    assert run_summary.refusal.diagnostic.provider == "openai"
    assert run_summary.refusal.diagnostic.source == "agent_output"


def test_empty_miniswe_trajectory_has_not_evaluated_refusal() -> None:
    run_summary = build_miniswe_run_summary(
        serialized_trajectory={"messages": []},
        duration_seconds=0,
        total_cost_usd=0,
        step_count=0,
    )

    assert run_summary.refusal.status == "not_evaluated"
    assert run_summary.refusal.diagnostic is None
