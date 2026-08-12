import json
import math
import os
import random
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

from latch_eval_tools.harness.run_summary import (
    CliHarnessAgentType,
    build_cli_run_summary,
)
from latch_eval_tools.harness.utils import (
    DEFAULT_DOCKER_IMAGE,
    ensure_docker_image,
    get_agent_workspace_dir,
    get_agent_workspace_mount_args,
    get_memory_limit_bytes,
    is_docker_container_oom_killed,
    is_docker_container_running,
    load_trajectory_identifier,
    prompt_with_suffix,
    render_packaged_prompt,
)
from latch_eval_tools.llm_refusal import detect_llm_refusal

REFUSAL_VERDICT_FILENAME = "refusal_verdict.json"


def _write_refusal_verdict(
    work_dir: Path,
    trajectory: list[dict[str, Any]],
    *,
    refusal_events: list[dict[str, Any]] | None = None,
    agent_error: str | None = None,
) -> None:
    # Detect refusals against the in-memory trajectory here (the harness has it
    # local) and persist the verdict, so consumers never re-read the trajectory.
    try:
        refusal = detect_llm_refusal(
            trajectory_data=trajectory,
            refusal_events_data=refusal_events,
            agent_error=agent_error,
        )
        payload = refusal.model_dump(mode="json") if refusal is not None else None
        (work_dir / REFUSAL_VERDICT_FILENAME).write_text(json.dumps(payload))
    except Exception as exc:
        print(f"failed to write refusal verdict: {exc}")


EVAL_TIMEOUT = 600
ANTHROPIC_ENV_KEYS = {"ANTHROPIC_API_KEY"}
OPENAI_ENV_KEYS = {"OPENAI_API_KEY", "CODEX_API_KEY"}
PI_ENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "FIREWORKS_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "XAI_API_KEY",
    "OPENROUTER_API_KEY",
}
GROK_ENV_KEYS = {"XAI_API_KEY"}

# pi ships no built-in OpenRouter provider, so we register a custom
# openai-completions provider in /root/.pi/agent/models.json. The pi state dir
# (work_dir/.pi) is bind-mounted to /root/.pi, so writing the file on the host
# lands it where pi looks. Notes on the fields:
#   - cost is per-MILLION tokens (OpenRouter list price: $3 / $15, $0.30 cache read).
#   - contextWindow/maxTokens must be explicit; pi defaults to 128k/16k and would
#     otherwise truncate K3's 1M window and long reasoning. (maxTokens passthrough
#     for openai-completions requires pi >= 0.80.3, issue #5595.)
#   - compat.thinkingFormat "reasoning_effort" sends the top-level reasoning_effort
#     field Moonshot documents for K3; K3 only supports the "max" level, so every
#     pi thinking level maps to "max".
#   - compat.supportsUsageInStreaming keeps stream_options.include_usage on so
#     usage (input/output/cache tokens) and cost are reported back.
OPENROUTER_PROVIDER_NAME = "openrouter"
OPENROUTER_PROVIDER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_CONFIGS: dict[str, dict] = {
    "openrouter/moonshotai/kimi-k3": {
        "id": "moonshotai/kimi-k3",
        "name": "Kimi K3",
        "reasoning": True,
        "input": ["text", "image"],
        "contextWindow": 1048576,
        "maxTokens": 131072,
        "cost": {"input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 0},
        "thinkingLevelMap": {
            "low": "max",
            "medium": "max",
            "high": "max",
            "xhigh": "max",
            "max": "max",
        },
        "compat": {
            "thinkingFormat": "reasoning_effort",
            "supportsReasoningEffort": True,
            "supportsUsageInStreaming": True,
        },
    },
    # DeepSeek-V4-Flash-0731: 1M context, reasoning_effort supports low/high/max
    # (no medium/xhigh at the API), and DeepSeek recommends max output 384K tokens
    # at the high/max effort levels. We pin every pi thinking level to "max" via
    # thinkingLevelMap so the model always runs at its top reasoning effort.
    # Cost is OpenRouter list price ($0.14 / $0.28 per 1M, $0.028 cache read;
    # Cloudflare provider).
    "openrouter/deepseek/deepseek-v4-flash-0731": {
        "id": "deepseek/deepseek-v4-flash-0731",
        "name": "DeepSeek V4 Flash",
        "reasoning": True,
        "input": ["text"],
        "contextWindow": 1048576,
        "maxTokens": 393216,
        "cost": {"input": 0.14, "output": 0.28, "cacheRead": 0.028, "cacheWrite": 0},
        "thinkingLevelMap": {
            "low": "max",
            "medium": "max",
            "high": "max",
            "xhigh": "max",
            "max": "max",
        },
        "compat": {
            "thinkingFormat": "reasoning_effort",
            "supportsReasoningEffort": True,
            "supportsUsageInStreaming": True,
        },
    },
}


def _write_pi_openrouter_models_json(work_dir: Path, model_name: str) -> None:
    if model_name not in OPENROUTER_MODEL_CONFIGS:
        raise ValueError(
            f"No pi OpenRouter model config registered for {model_name!r}; "
            "add it to OPENROUTER_MODEL_CONFIGS in _cli_runner.py"
        )
    models_json = {
        "providers": {
            OPENROUTER_PROVIDER_NAME: {
                "baseUrl": OPENROUTER_PROVIDER_BASE_URL,
                "apiKey": "$OPENROUTER_API_KEY",
                "api": "openai-completions",
                "models": [OPENROUTER_MODEL_CONFIGS[model_name]],
            }
        }
    }
    models_path = work_dir / AGENT_STATE_DIRS["pi"] / "agent" / "models.json"
    models_path.parent.mkdir(parents=True, exist_ok=True)
    models_path.write_text(json.dumps(models_json, indent=2), encoding="utf-8")


OOM_EXIT_CODE = 137
MAX_OOM_RESTARTS = 10

# claude-code runs one-shot (`--print`): when the model ends its turn the
# process exits, ending the session and killing any background tasks it
# started. Agents routinely end their turn to "wait" for a long-running job,
# expecting to be re-invoked -- nothing does that, so the run terminates
# immediately with no answer. When claude-code exits cleanly without an answer
# file, re-invoke `claude --resume` with a nudge (up to this many times) so the
# agent notices the unfinished work and blocks on it synchronously instead.
MAX_CLAUDECODE_ANSWER_RESUMES = 30
CLAUDECODE_RESUME_NUDGE = (
    "You ended your turn without submitting a final answer, so the session was"
    " about to end. Any background process you started has been killed and is no"
    " longer running. Check the current state on disk, finish any unfinished"
    " work using a synchronous foreground command that blocks until the job"
    " completes, then submit your final answer. Do not end your turn again"
    " until the work is done and the answer has been written."
)
AGENT_STATE_DIRS = {
    "claudecode": ".claude",
    "openaicodex": ".codex",
    "pi": ".pi",
    "grokbuild": ".grok",
}
# By default the host state dir is bind-mounted onto /root/<state_dir_name>.
# grok is the exception: its ENTIRE install (binary in ~/.grok/downloads,
# symlinked via ~/.grok/bin) lives under ~/.grok, so mounting the state dir onto
# /root/.grok would shadow the binary and break `grok`. Mount onto the sessions
# subdir instead -- persists sessions across container recreation without
# hiding the install. (Verified: an empty mount over /root/.grok makes `grok`
# vanish from PATH; a mount over /root/.grok/sessions leaves it intact.)
AGENT_CONTAINER_STATE_MOUNTS = {
    "grokbuild": "/root/.grok/sessions",
}
AGENT_IDENTIFIER_KEYS = {
    "claudecode": "session_id",
    "openaicodex": "thread_id",
    "pi": "id",
    # grok's final `end` event carries `sessionId` at the top level, so the flat
    # load_trajectory_identifier() finds it and resume works.
    "grokbuild": "sessionId",
}
PI_IGNORED_EVENT_TYPES = {"message_update", "tool_execution_update"}
PI_TOOL_TIMEOUT_EXTENSION_RELATIVE_PATH = Path(".latch_eval_tools", "tool_timeout.js")
PI_TOOL_TIMEOUT_EXTENSION_CONTAINER_PATH = (
    f"/workspace/{PI_TOOL_TIMEOUT_EXTENSION_RELATIVE_PATH}"
)
PROVIDER_RETRYABLE_STATUS_CODES = frozenset(
    {408, 409, 425, 429, 500, 502, 503, 504, 520, 529}
)
PROVIDER_CAPACITY_STATUS_CODES = frozenset({429, 529})
PROVIDER_MAX_RESUMES = 1
PROVIDER_CAPACITY_FALLBACK_SECONDS = 60.0
PROVIDER_CAPACITY_JITTER_SECONDS = 15.0
PROVIDER_TRANSPORT_FALLBACK_SECONDS = 5.0
PROVIDER_TRANSPORT_JITTER_SECONDS = 5.0
PROVIDER_HINT_JITTER_SECONDS = 5.0
PI_ASSISTANT_EVENT_TYPES = frozenset({"message", "message_end"})


@dataclass(frozen=True)
class ProviderFailure:
    status_code: int
    retry_after_seconds: float | None

    @property
    def retryable(self) -> bool:
        return self.status_code in PROVIDER_RETRYABLE_STATUS_CODES

    @property
    def capacity_limited(self) -> bool:
        return self.status_code in PROVIDER_CAPACITY_STATUS_CODES

    @property
    def error_code(self) -> str:
        if self.status_code == 429:
            return "rate_limit"
        if self.status_code == 529:
            return "overloaded"
        if self.status_code >= 500:
            return "server_error"
        return f"http_{self.status_code}"


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_nonnegative_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _optional_nonnegative_decimal(value: object) -> float | None:
    if isinstance(value, str):
        try:
            return _optional_nonnegative_float(float(value))
        except ValueError:
            return None
    return _optional_nonnegative_float(value)


def _json_object(value: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _provider_error_payload(
    error_message: str,
) -> tuple[int | None, dict[str, object]] | None:
    json_start = error_message.find("{")
    if json_start < 0:
        return None
    prefix = error_message[:json_start].strip().removesuffix(":").strip()
    prefix_status = int(prefix) if prefix.isdigit() else None
    payload = _json_object(error_message[json_start:])
    if payload is None:
        return None
    return prefix_status, payload


def _payload_error(payload: dict[str, object]) -> dict[str, object] | None:
    value = payload.get("error")
    if not isinstance(value, dict):
        return None
    return value


def _nested_google_payload(
    payload: dict[str, object],
) -> dict[str, object] | None:
    outer_error = _payload_error(payload)
    if outer_error is None:
        return None
    message = outer_error.get("message")
    if not isinstance(message, str):
        return None
    return _json_object(message)


def _provider_status(
    prefix_status: int | None,
    payload: dict[str, object],
) -> int | None:
    if prefix_status is not None:
        return prefix_status
    status = _optional_int(payload.get("code"))
    if status is not None:
        return status
    error_payload = _payload_error(payload)
    if error_payload is None:
        return None
    return _optional_int(error_payload.get("code"))


def _grpc_retry_delay_seconds(value: object) -> float | None:
    if not isinstance(value, str) or not value.endswith("s"):
        return None
    try:
        return _optional_nonnegative_float(float(value[:-1]))
    except ValueError:
        return None


def _google_retry_hint_seconds(
    nested_payload: dict[str, object] | None,
) -> float | None:
    if nested_payload is None:
        return None
    nested_error = _payload_error(nested_payload)
    if nested_error is None:
        return None
    details = nested_error.get("details")
    if not isinstance(details, list):
        return None
    for detail in details:
        if not isinstance(detail, dict):
            continue
        if detail.get("@type") != "type.googleapis.com/google.rpc.RetryInfo":
            continue
        return _grpc_retry_delay_seconds(detail.get("retryDelay"))
    return None


def _openrouter_retry_hint_seconds(payload: dict[str, object]) -> float | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None

    retry_after = _optional_nonnegative_float(metadata.get("retry_after_seconds"))
    headers = metadata.get("headers")
    if not isinstance(headers, dict):
        return retry_after

    header_retry_after = _optional_nonnegative_decimal(headers.get("Retry-After"))
    if header_retry_after is not None:
        retry_after = max(retry_after or 0.0, header_retry_after)

    reset_milliseconds = _optional_nonnegative_decimal(headers.get("X-RateLimit-Reset"))
    if reset_milliseconds is not None:
        reset_delay = max((reset_milliseconds / 1000.0) - time.time(), 0.0)
        retry_after = max(retry_after or 0.0, reset_delay)
    return retry_after


def _pi_provider_failure(attempt_events: list[dict]) -> ProviderFailure | None:
    for event in reversed(attempt_events):
        if event.get("type") not in PI_ASSISTANT_EVENT_TYPES:
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if message.get("stopReason") != "error":
            return None
        error_message = message.get("errorMessage")
        if not isinstance(error_message, str):
            return None
        parsed = _provider_error_payload(error_message)
        if parsed is None:
            return None
        prefix_status, payload = parsed
        status_code = _provider_status(prefix_status, payload)
        if status_code is None:
            return None
        nested_payload = _nested_google_payload(payload)
        retry_hints = [
            hint
            for hint in (
                _openrouter_retry_hint_seconds(payload),
                _google_retry_hint_seconds(nested_payload),
            )
            if hint is not None
        ]
        return ProviderFailure(
            status_code=status_code,
            retry_after_seconds=max(retry_hints, default=None),
        )
    return None


def _claudecode_provider_failure(
    attempt_events: list[dict],
    *,
    include_inflight_retry: bool,
) -> ProviderFailure | None:
    for result_index in range(len(attempt_events) - 1, -1, -1):
        result = attempt_events[result_index]
        if result.get("type") != "result":
            continue
        if result.get("terminal_reason") != "api_error":
            return None
        status_code = _optional_int(result.get("api_error_status"))
        if status_code is None:
            return None

        retry_after_seconds: float | None = None
        for event in reversed(attempt_events[:result_index]):
            if event.get("type") in {"assistant", "result"}:
                break
            if event.get("type") != "system" or event.get("subtype") != "api_retry":
                continue
            if _optional_int(event.get("error_status")) != status_code:
                continue
            retry_delay_ms = _optional_nonnegative_float(event.get("retry_delay_ms"))
            if retry_delay_ms is not None:
                retry_after_seconds = retry_delay_ms / 1000.0
            break

        return ProviderFailure(
            status_code=status_code,
            retry_after_seconds=retry_after_seconds,
        )

    if include_inflight_retry:
        for event in reversed(attempt_events):
            if event.get("type") == "assistant":
                return None
            if event.get("type") != "system" or event.get("subtype") != "api_retry":
                continue
            status_code = _optional_int(event.get("error_status"))
            if status_code is None:
                return None
            retry_delay_ms = _optional_nonnegative_float(event.get("retry_delay_ms"))
            return ProviderFailure(
                status_code=status_code,
                retry_after_seconds=(
                    retry_delay_ms / 1000.0 if retry_delay_ms is not None else None
                ),
            )
    return None


def classify_terminal_provider_failure(
    agent_type: str,
    attempt_events: list[dict],
    *,
    include_inflight_retry: bool = False,
) -> ProviderFailure | None:
    if agent_type == "claudecode":
        return _claudecode_provider_failure(
            attempt_events,
            include_inflight_retry=include_inflight_retry,
        )
    if agent_type == "pi":
        return _pi_provider_failure(attempt_events)
    return None


def provider_retry_delay_seconds(failure: ProviderFailure) -> float:
    if not failure.retryable:
        raise ValueError("provider failure is not retryable")
    if failure.retry_after_seconds is not None:
        return failure.retry_after_seconds + random.uniform(
            0.0, PROVIDER_HINT_JITTER_SECONDS
        )
    if failure.capacity_limited:
        return PROVIDER_CAPACITY_FALLBACK_SECONDS + random.uniform(
            0.0, PROVIDER_CAPACITY_JITTER_SECONDS
        )
    return PROVIDER_TRANSPORT_FALLBACK_SECONDS + random.uniform(
        0.0, PROVIDER_TRANSPORT_JITTER_SECONDS
    )


def teardown_container(container_name: str) -> None:
    try:
        remove_result = subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if remove_result.returncode != 0:
            stderr = remove_result.stderr.strip()
            if not stderr or "No such container" not in stderr:
                print(f"Failed to remove container {container_name}: {stderr}")
    except Exception as exc:
        print(f"Failed to remove container {container_name}: {exc}")


def _build_agent_command(
    agent_type: str,
    cli_command: list[str],
    model_name: str | None,
    model_map: dict[str, str] | None,
    claude_code_extra_args: list[str] | None,
    resume_identifier: str | None = None,
    system_prompt: str | None = None,
    prompt_text: str | None = None,
) -> list[str]:
    if agent_type == "claudecode":
        agent_cmd = list(cli_command)
        if resume_identifier is not None:
            agent_cmd.extend(["--resume", resume_identifier])
        agent_cmd.extend(
            [
                "--print",
                "--dangerously-skip-permissions",
                "--effort",
                "max",
                "--verbose",
                "--output-format",
                "stream-json",
                "--include-partial-messages",
                "--thinking-display",
                "summarized",
            ]
        )
        if claude_code_extra_args:
            agent_cmd.extend(claude_code_extra_args)
        if system_prompt not in (None, ""):
            agent_cmd.extend(["--system-prompt", system_prompt])
    elif agent_type == "openaicodex":
        agent_cmd = list(cli_command)
        if resume_identifier is not None:
            agent_cmd.append("resume")
        agent_cmd.extend(
            [
                "--dangerously-bypass-approvals-and-sandbox",
                "--skip-git-repo-check",
                "--json",
                "-c",
                'model_reasoning_effort="xhigh"',
                "-c",
                'model_reasoning_summary="detailed"',
                "-c",
                "hide_agent_reasoning=false",
                "-c",
                "show_raw_agent_reasoning=true",
                "-c",
                "model_supports_reasoning_summaries=true",
            ]
        )
    elif agent_type == "pi":
        agent_cmd = list(cli_command)
        agent_cmd.extend(["--mode", "json", "--print"])
        if resume_identifier is not None:
            agent_cmd.extend(["--session", resume_identifier])
        agent_cmd.extend(["--thinking", "xhigh"])
        agent_cmd.extend(["--extension", PI_TOOL_TIMEOUT_EXTENSION_CONTAINER_PATH])
        if system_prompt not in (None, ""):
            agent_cmd.extend(["--system-prompt", system_prompt])
    elif agent_type == "grokbuild":
        agent_cmd = list(cli_command)
        agent_cmd.extend(
            [
                "--no-auto-update",
                "--no-alt-screen",
                "--always-approve",
                "--cwd",
                "/workspace",
                "--output-format",
                "streaming-json",
            ]
        )
        if resume_identifier is not None:
            agent_cmd.extend(["--resume", resume_identifier])
        if system_prompt not in (None, ""):
            agent_cmd.extend(["--system-prompt", system_prompt])
        agent_cmd.extend(["-p", prompt_text if prompt_text is not None else ""])
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    if model_name and model_map:
        mapped_model = model_map.get(model_name, model_name)
        agent_cmd.extend(["--model", mapped_model])
    elif model_name:
        agent_cmd.extend(["--model", model_name])
    if agent_type == "openaicodex" and resume_identifier is not None:
        agent_cmd.append(resume_identifier)
    return agent_cmd


def _create_cli_container(
    container_name: str,
    agent_type: str,
    work_dir: Path,
    agent_dir: Path,
    env_flags: list[str],
    docker_image: str,
    memory_limit_bytes: int,
) -> str:
    state_dir_name = AGENT_STATE_DIRS.get(agent_type)
    if state_dir_name is None:
        raise ValueError(f"Unknown agent type for state dir: {agent_type}")

    agent_state_dir = work_dir / state_dir_name
    agent_state_dir.mkdir(parents=True, exist_ok=True)
    container_state_mount = AGENT_CONTAINER_STATE_MOUNTS.get(
        agent_type, f"/root/{state_dir_name}"
    )
    subprocess.run(
        [
            "docker",
            "create",
            "--name",
            container_name,
            "-i",
            "--memory",
            str(memory_limit_bytes),
            "--memory-swap",
            str(memory_limit_bytes),
            *get_agent_workspace_mount_args(agent_dir),
            "-v",
            f"{agent_state_dir}:{container_state_mount}",
            "-w",
            "/workspace",
            *env_flags,
            docker_image,
            "sleep",
            "infinity",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return container_state_mount


def _start_cli_container(container_name: str) -> None:
    try:
        result = subprocess.run(
            ["docker", "start", container_name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            print(f"Failed to start container {container_name}: {stderr}")
        assert result.returncode == 0, f"Failed to start container {container_name}"
    except Exception as e:
        print(f"Error starting container {container_name}: {e}")
        raise e


def _extract_last_message(trajectory: list[dict], agent_type: str) -> str:
    """Best-effort extract the agent's last assistant message from a CLI
    trajectory. Used by completion mode to surface a non-null agent_answer.
    Trajectory shape varies per CLI (claudecode/codex stream different events
    than pi), so this is permissive — any string that looks like assistant
    output works.
    """
    if agent_type == "grokbuild":
        # grok streams assistant text as flat {"type":"text","data":"..."} chunks,
        # emitting a fresh run of them after each tool call. Return the last
        # contiguous run (the final narration), not every chunk concatenated.
        segments: list[str] = []
        current: list[str] = []
        for event in trajectory:
            if event.get("type") == "text" and isinstance(event.get("data"), str):
                current.append(event["data"])
            elif current:
                segments.append("".join(current))
                current = []
        if current:
            segments.append("".join(current))
        for segment in reversed(segments):
            if segment.strip():
                return segment.strip()
        return ""

    for event in reversed(trajectory):
        if not isinstance(event, dict):
            continue
        message = event.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str) and content.strip() != "":
                return content
            if isinstance(content, list):
                # Anthropic-style content blocks: [{"type":"text","text":"..."}]
                for block in reversed(content):
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                        and block["text"].strip() != ""
                    ):
                        return block["text"]
        # codex/claudecode sometimes emit top-level {"type":"text","text":...}
        if event.get("type") == "text":
            text = event.get("text")
            if isinstance(text, str) and text.strip() != "":
                return text
    return ""


def _pi_clean_exit_needs_resume(attempt_events: list[dict]) -> bool:
    for event in reversed(attempt_events):
        if event.get("type") == "compaction_end":
            return event.get("aborted") is not True

        message = event.get("message") or {}
        if message.get("role") == "assistant" and message.get("stopReason") == "length":
            return True

    return False


def _iter_jsonl_objects(source: Path) -> Iterator[dict[str, Any]]:
    try:
        with source.open(encoding="utf-8") as source_file:
            for line in source_file:
                stripped = line.strip()
                if stripped == "":
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event
    except (OSError, UnicodeDecodeError) as exc:
        print(f"Failed to read harness sidecar {source}: {exc}")


def _read_codex_sidecar_events(
    work_dir: Path,
    trajectory: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    thread_id = None
    for event in trajectory:
        value = event.get("thread_id")
        if isinstance(value, str) and value:
            thread_id = value
            break
    if thread_id is None:
        return None

    codex_dir = work_dir / AGENT_STATE_DIRS["openaicodex"]
    if not codex_dir.exists():
        return None

    matched_source = False
    events: list[dict[str, Any]] = []
    for source in sorted(codex_dir.rglob("*")):
        if source.is_dir() or source.is_symlink() or thread_id not in source.name:
            continue
        matched_source = True
        for event in _iter_jsonl_objects(source):
            event_type = event.get("type")
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            if event_type == "response_item" and payload.get("type") in {
                "function_call",
                "reasoning",
            }:
                events.append(event)
            elif event_type == "event_msg" and payload.get("type") == "token_count":
                events.append(event)
    return events if matched_source else None


def _read_pi_refusal_events(
    work_dir: Path,
) -> list[dict[str, Any]] | None:
    source = work_dir / AGENT_STATE_DIRS["pi"] / "refusal_events.jsonl"
    if not source.exists() or source.is_dir() or source.is_symlink():
        return None
    return list(_iter_jsonl_objects(source))


def _append_codex_sidecar_reasoning(
    trajectory: list[dict[str, Any]],
    sidecar_events: list[dict[str, Any]],
) -> int:

    existing_reasoning_ids = {
        event.get("payload", {}).get("id")
        for event in trajectory
        if (
            isinstance(event, dict)
            and event.get("type") == "response_item"
            and isinstance(event.get("payload"), dict)
            and event["payload"].get("type") == "reasoning"
        )
    }

    appended = 0
    for event in sidecar_events:
        if event.get("type") != "response_item":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "reasoning":
            continue
        payload_id = payload.get("id")
        if payload_id in existing_reasoning_ids:
            continue
        trajectory.append(
            {
                "type": "response_item",
                "source": "codex_sidecar",
                "timestamp": event.get("timestamp"),
                "payload": payload,
            }
        )
        existing_reasoning_ids.add(payload_id)
        appended += 1

    return appended


def _run_cli_agent(
    agent_type: CliHarnessAgentType,
    cli_command: list[str],
    task_prompt: str,
    work_dir: Path,
    model_name: str | None = None,
    eval_timeout: int = EVAL_TIMEOUT,
    model_map: dict[str, str] | None = None,
    claude_code_extra_args: list[str] | None = None,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    memory_limit_bytes: int | None = None,
    system_prompt: str | None = None,
    prompt_suffix: str | None = None,
    completion: bool = False,
    benchmark: bool = False,
    operation_timeout: int = 0,
) -> dict:
    agent_log_file = work_dir / "agent_output.log"
    if agent_log_file.exists():
        agent_log_file.unlink()
    enhanced_prompt = prompt_with_suffix(task_prompt, prompt_suffix)

    env = os.environ.copy()

    if agent_type == "openaicodex":
        if "CODEX_API_KEY" not in env and "OPENAI_API_KEY" in env:
            env["CODEX_API_KEY"] = env["OPENAI_API_KEY"]

    if not docker_image:
        raise ValueError("docker_image is required for CLI harnesses")

    ensure_docker_image(docker_image)
    agent_dir = get_agent_workspace_dir(work_dir)
    if agent_type == "pi":
        extension_path = agent_dir / PI_TOOL_TIMEOUT_EXTENSION_RELATIVE_PATH
        extension_path.parent.mkdir(parents=True, exist_ok=True)
        extension_source = (
            files("latch_eval_tools")
            .joinpath("pi_extensions", PI_TOOL_TIMEOUT_EXTENSION_RELATIVE_PATH.name)
            .read_text(encoding="utf-8")
        )
        extension_path.write_text(extension_source, encoding="utf-8")
    env_flags: list[str] = []
    ENV_KEYS = {}
    if agent_type == "claudecode":
        ENV_KEYS = ANTHROPIC_ENV_KEYS
    elif agent_type == "openaicodex":
        ENV_KEYS = OPENAI_ENV_KEYS
    elif agent_type == "pi":
        ENV_KEYS = PI_ENV_KEYS
    elif agent_type == "grokbuild":
        ENV_KEYS = GROK_ENV_KEYS
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    extra_env_keys = set(json.loads(env.get("EXTRA_ENV_KEYS") or "[]"))
    ENV_KEYS = ENV_KEYS | extra_env_keys
    for key in ENV_KEYS:
        value = env.get(key)
        if value:
            env_flags.extend(["-e", f"{key}={value}"])
    if agent_type == "claudecode":
        bash_timeout_ms = eval_timeout * 1000
        env_flags.extend(
            [
                "-e",
                f"BASH_DEFAULT_TIMEOUT_MS={bash_timeout_ms}",
                "-e",
                f"BASH_MAX_TIMEOUT_MS={bash_timeout_ms}",
            ]
        )
    if agent_type == "pi":
        env_flags.extend(
            [
                "-e",
                "PI_SKIP_VERSION_CHECK=1",
                "-e",
                "PI_TELEMETRY=0",
                "-e",
                "NODE_OPTIONS=--max-old-space-size=8192",
            ]
        )
        if operation_timeout > 0:
            env_flags.extend(
                ["-e", f"PI_BASH_DEFAULT_TIMEOUT_SECONDS={operation_timeout}"]
            )
    if memory_limit_bytes is None:
        memory_limit_bytes = get_memory_limit_bytes()
    container_name = f"eval-{agent_type}-{uuid.uuid4().hex[:8]}"

    agent_start_time = time.time()
    agent_finished_at = agent_start_time
    timed_out = False
    agent_error: Exception | None = None
    trajectory: list[dict] = []
    trajectory_file = work_dir / "trajectory.json"
    trajectory_file.write_text(json.dumps(trajectory, indent=2))
    eval_answer_file = agent_dir / "eval_answer.json"
    finished_file = agent_dir / "finished.txt"
    oom_detected = False
    oom_restarts = 0
    provider_resumes = 0
    last_provider_failure: ProviderFailure | None = None

    trajectory_lock = threading.Lock()

    def persist_trajectory():
        with trajectory_lock:
            trajectory_file.write_text(json.dumps(trajectory, indent=2))

    try:
        container_state_mount = _create_cli_container(
            container_name=container_name,
            agent_type=agent_type,
            work_dir=work_dir,
            agent_dir=agent_dir,
            env_flags=env_flags,
            docker_image=docker_image,
            memory_limit_bytes=memory_limit_bytes,
        )
        # Register the custom OpenRouter provider into the (bind-mounted) pi state
        # dir. Written on the host so it survives container recreation on OOM.
        if agent_type == "pi" and model_name and model_name.startswith("openrouter/"):
            _write_pi_openrouter_models_json(work_dir, model_name)
        _start_cli_container(container_name)
        deadline = time.time() + eval_timeout

        with open(agent_log_file, "w") as log_file:
            agent_start_time = time.time()
            prompt_text = enhanced_prompt
            resume_identifier: str | None = None
            last_return_code: int | None = None
            claudecode_answer_resumes = 0

            while True:
                remaining_timeout = deadline - time.time()
                if remaining_timeout <= 0:
                    timed_out = True
                    log_file.write(
                        f"\n\nAgent timed out after {eval_timeout} seconds\n"
                    )
                    log_file.flush()
                    break

                # Only carry provider evidence from the final failed attempt.
                # A recovered rate limit must not relabel a later unrelated error.
                last_provider_failure = None
                attempt_start_index = len(trajectory)
                agent_cmd = _build_agent_command(
                    agent_type=agent_type,
                    cli_command=cli_command,
                    model_name=model_name,
                    model_map=model_map,
                    claude_code_extra_args=claude_code_extra_args,
                    resume_identifier=resume_identifier,
                    system_prompt=system_prompt,
                    prompt_text=prompt_text,
                )

                process = subprocess.Popen(
                    ["docker", "exec", "-i", container_name, *agent_cmd],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(agent_dir),
                    env=env,
                    text=True,
                    bufsize=1,
                )

                stderr_header_written = False
                stderr_lock = threading.Lock()

                def stream_stdout():
                    if process.stdout is None:
                        return
                    try:
                        for line in process.stdout:
                            if agent_type != "pi":
                                log_file.write(line)
                                log_file.flush()

                            stripped = line.strip()
                            if not stripped:
                                continue
                            try:
                                event = json.loads(stripped)
                                if not isinstance(event, dict):
                                    continue
                                if (
                                    agent_type == "pi"
                                    and event.get("type") in PI_IGNORED_EVENT_TYPES
                                ):
                                    continue
                                with trajectory_lock:
                                    trajectory.append(event)
                                persist_trajectory()
                            except json.JSONDecodeError:
                                print(f"Warning: Failed to parse JSON: {stripped}")
                    except ValueError:
                        pass

                def stream_stderr():
                    nonlocal stderr_header_written
                    if process.stderr is None:
                        return
                    try:
                        for line in process.stderr:
                            with stderr_lock:
                                if not stderr_header_written:
                                    log_file.write("\n\nSTDERR:\n")
                                    stderr_header_written = True
                                log_file.write(line)
                                log_file.flush()
                    except ValueError:
                        pass

                stdout_thread = threading.Thread(target=stream_stdout, daemon=True)
                stderr_thread = threading.Thread(target=stream_stderr, daemon=True)
                stdout_thread.start()
                stderr_thread.start()

                if process.stdin is not None:
                    process.stdin.write(prompt_text)
                    process.stdin.close()

                timed_out_attempt = False
                answer_submitted = False
                try:
                    while process.poll() is None:
                        now = time.time()
                        remaining_timeout = deadline - now
                        if remaining_timeout <= 0:
                            raise subprocess.TimeoutExpired(
                                process.args, remaining_timeout
                            )

                        try:
                            if agent_type == "pi":
                                if completion and finished_file.exists():
                                    answer_submitted = True
                                elif not completion and eval_answer_file.exists():
                                    json.loads(eval_answer_file.read_text())
                                    answer_submitted = True
                                if answer_submitted:
                                    process.terminate()
                                    try:
                                        process.wait(timeout=10)
                                    except subprocess.TimeoutExpired:
                                        process.kill()
                                        process.wait()
                                    break
                        except (json.JSONDecodeError, OSError):
                            pass

                        time.sleep(min(1, remaining_timeout))
                except subprocess.TimeoutExpired:
                    timed_out_attempt = True
                    process.kill()
                    process.wait()

                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)
                last_return_code = process.returncode
                attempt_events = trajectory[attempt_start_index:]
                if answer_submitted:
                    log_file.write("\n\nDetected eval_answer.json, stopping agent\n")
                    log_file.flush()
                    break
                provider_failure = classify_terminal_provider_failure(
                    agent_type,
                    attempt_events,
                    include_inflight_retry=timed_out_attempt,
                )
                if timed_out_attempt:
                    last_provider_failure = provider_failure
                    log_file.write(
                        f"\n\nAgent timed out after {eval_timeout} seconds\n"
                    )
                    log_file.flush()
                    timed_out = True
                    break

                if last_return_code == 0 and (
                    eval_answer_file.exists() or finished_file.exists()
                ):
                    break

                if provider_failure is not None:
                    last_provider_failure = provider_failure
                    if provider_failure.retryable:
                        persist_trajectory()
                        candidate_resume_identifier = load_trajectory_identifier(
                            trajectory_file,
                            AGENT_IDENTIFIER_KEYS[agent_type],
                        )
                        delay = provider_retry_delay_seconds(provider_failure)
                        retry_fits_deadline = delay < deadline - time.time()
                        if (
                            provider_resumes < PROVIDER_MAX_RESUMES
                            and candidate_resume_identifier is not None
                            and retry_fits_deadline
                            and is_docker_container_running(container_name)
                        ):
                            provider_resumes += 1
                            resume_identifier = candidate_resume_identifier
                            log_file.write(
                                "\n\n[Provider retry "
                                f"{provider_resumes}/{PROVIDER_MAX_RESUMES}] "
                                f"waiting {delay:.1f}s before resuming session "
                                f"{resume_identifier}\n"
                            )
                            log_file.flush()
                            time.sleep(delay)
                            prompt_text = "Continue."
                            continue
                    # Do not bypass provider retry limits through the generic
                    # clean-exit or OOM resume paths below.
                    break

                if last_return_code == 0:
                    if (
                        agent_type == "pi"
                        and not eval_answer_file.exists()
                        and _pi_clean_exit_needs_resume(attempt_events)
                    ):
                        persist_trajectory()
                        resume_identifier = load_trajectory_identifier(
                            trajectory_file,
                            AGENT_IDENTIFIER_KEYS["pi"],
                        )
                        if resume_identifier is None:
                            agent_error = RuntimeError(
                                "pi compacted or hit length before emitting "
                                "a session id"
                            )
                            break

                        log_file.write(
                            "\n\nPi compacted or hit length before writing "
                            "eval_answer.json; resuming session "
                            f"{resume_identifier}\n"
                        )
                        log_file.flush()
                        prompt_text = "Continue."
                        continue
                    if agent_type in ("claudecode", "grokbuild"):
                        if benchmark:
                            answer_present = (
                                finished_file.exists()
                                if completion
                                else eval_answer_file.exists()
                            )
                            if (
                                not answer_present
                                and claudecode_answer_resumes
                                < MAX_CLAUDECODE_ANSWER_RESUMES
                            ):
                                persist_trajectory()
                                resume_identifier = load_trajectory_identifier(
                                    trajectory_file,
                                    AGENT_IDENTIFIER_KEYS[agent_type],
                                )
                                if resume_identifier is not None:
                                    claudecode_answer_resumes += 1
                                    log_file.write(
                                        f"\n\n{agent_type} ended its turn without a "
                                        "final answer; resuming session "
                                        f"{resume_identifier} (resume "
                                        f"{claudecode_answer_resumes}/"
                                        f"{MAX_CLAUDECODE_ANSWER_RESUMES})\n"
                                    )
                                    log_file.flush()
                                    prompt_text = CLAUDECODE_RESUME_NUDGE
                                    continue
                    break

                container_running = is_docker_container_running(container_name)
                container_oom_killed = is_docker_container_oom_killed(container_name)
                attempt_hit_oom = (
                    last_return_code == OOM_EXIT_CODE or container_oom_killed
                )
                if not attempt_hit_oom:
                    agent_error = RuntimeError(
                        f"{agent_type} exited with code {last_return_code}"
                    )
                    break

                oom_detected = True
                if oom_restarts >= MAX_OOM_RESTARTS:
                    agent_error = RuntimeError(
                        f"{agent_type} exceeded max OOM restarts ({MAX_OOM_RESTARTS})"
                    )
                    log_file.write(
                        f"\n\nExceeded max OOM restarts ({MAX_OOM_RESTARTS})\n"
                    )
                    log_file.flush()
                    break

                identifier_key = AGENT_IDENTIFIER_KEYS.get(agent_type)
                if identifier_key is None:
                    raise ValueError(
                        f"Unknown agent type for resume identifier: {agent_type}"
                    )

                persist_trajectory()
                resume_identifier = load_trajectory_identifier(
                    trajectory_file,
                    identifier_key,
                )
                if resume_identifier is None:
                    agent_error = RuntimeError(
                        f"{agent_type} hit OOM before emitting {identifier_key}"
                    )
                    break

                if container_running:
                    container_action = (
                        "The execution container stayed alive and the agent process "
                        "is being resumed in place."
                    )
                else:
                    subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    container_state_mount = _create_cli_container(
                        container_name=container_name,
                        agent_type=agent_type,
                        work_dir=work_dir,
                        agent_dir=agent_dir,
                        env_flags=env_flags,
                        docker_image=docker_image,
                        memory_limit_bytes=memory_limit_bytes,
                    )
                    _start_cli_container(container_name)
                    container_action = (
                        "The execution container was restarted before the session "
                        "was resumed."
                    )

                oom_restarts += 1
                log_file.write(
                    f"\n\n[OOM restart {oom_restarts}/{MAX_OOM_RESTARTS}]\n"
                    f"{container_action}\n"
                )
                log_file.flush()
                prompt_text = render_packaged_prompt(
                    "oom_restart.md",
                    container_action=container_action,
                    state_dir=container_state_mount,
                )

    except Exception as e:
        agent_error = e
        with open(agent_log_file, "a") as f:
            f.write(f"\nError running {agent_type}: {e}")
    finally:
        agent_finished_at = time.time()
        teardown_container(container_name)

    duration = agent_finished_at - agent_start_time
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Agent output saved to: {agent_log_file}"
    )

    codex_sidecar_events: list[dict[str, Any]] | None = None
    pi_refusal_events: list[dict[str, Any]] | None = None
    if agent_type == "openaicodex" and trajectory:
        codex_sidecar_events = _read_codex_sidecar_events(work_dir, trajectory)
        appended_reasoning = (
            _append_codex_sidecar_reasoning(trajectory, codex_sidecar_events)
            if codex_sidecar_events is not None
            else 0
        )
        if appended_reasoning > 0:
            print(
                "Appended "
                f"{appended_reasoning} Codex reasoning item(s) from sidecar to trajectory"
            )
    elif agent_type == "pi":
        pi_refusal_events = _read_pi_refusal_events(work_dir)

    if trajectory:
        persist_trajectory()
        print(f"Trajectory saved to: {trajectory_file}")

    eval_answer_file = agent_dir / "eval_answer.json"
    agent_answer = None
    error_details = None

    def _log_tail() -> str:
        if not agent_log_file.exists():
            return ""
        return agent_log_file.read_text()[-1000:]

    if completion:
        # completion mode has no answer file. Surface the agent's last
        # message (extracted from the streamed trajectory) so downstream
        # consumers have something more useful than ``null``.
        if not finished_file.exists():
            if timed_out:
                error_msg = "Agent timed out"
            elif agent_error is not None:
                error_msg = f"{type(agent_error).__name__}: {agent_error}"
            else:
                error_msg = "Agent did not create finished.txt"
            error_details = {
                "error": error_msg,
                "timed_out": timed_out,
                "log_tail": _log_tail(),
            }
            print(f"\nWarning: {error_msg}")
        else:
            last_message = _extract_last_message(trajectory, agent_type)
            agent_answer = {
                "last_message": last_message,
                "finished_file_contents": finished_file.read_text(),
            }
    elif not eval_answer_file.exists():
        if finished_file.exists():
            last_message = _extract_last_message(trajectory, agent_type)
            agent_answer = {
                "last_message": last_message,
                "finished_file_contents": finished_file.read_text(),
            }
        else:
            if timed_out:
                error_msg = "Agent timed out"
            elif agent_error is not None:
                error_msg = f"{type(agent_error).__name__}: {agent_error}"
            else:
                error_msg = (
                    "no final answer provided (either eval_answer.json or finished.txt)"
                )
            error_details = {
                "error": error_msg,
                "timed_out": timed_out,
                "log_tail": _log_tail(),
            }
            print(f"\nWarning: {error_msg}")
    else:
        try:
            agent_answer = json.loads(eval_answer_file.read_text())
        except json.JSONDecodeError as e:
            error_details = {
                "error": f"Failed to parse eval_answer.json: {e}",
                "file_contents": eval_answer_file.read_text()[:500],
            }
            print(f"\nWarning: Failed to parse eval_answer.json: {e}")

    if error_details is not None and last_provider_failure is not None:
        error_details["api_error_code"] = last_provider_failure.error_code
        error_details["api_error_status"] = last_provider_failure.status_code
        if last_provider_failure.retry_after_seconds is not None:
            error_details["retry_after_seconds"] = (
                last_provider_failure.retry_after_seconds
            )
        error_details["provider_retry_count"] = provider_resumes

    structured_agent_error = None
    if error_details is not None:
        error_value = error_details.get("error")
        if isinstance(error_value, str):
            structured_agent_error = error_value

    metadata = _extract_metadata(
        agent_type,
        trajectory,
        duration,
        model_name,
        timed_out,
        eval_timeout,
        error_details,
        oom_detected=oom_detected,
        oom_restarts=oom_restarts,
        memory_limit_bytes=memory_limit_bytes,
        codex_sidecar_events=codex_sidecar_events,
        refusal_events=pi_refusal_events,
        agent_error=structured_agent_error,
    )

    _write_refusal_verdict(
        work_dir,
        trajectory,
        refusal_events=pi_refusal_events,
        agent_error=structured_agent_error,
    )

    return {"answer": agent_answer, "metadata": metadata}


def _extract_metadata(
    agent_type: CliHarnessAgentType,
    trajectory: list[dict[str, Any]],
    duration: float,
    model_name: str | None,
    timed_out: bool,
    eval_timeout: int,
    error_details: dict[str, Any] | None,
    oom_detected: bool,
    oom_restarts: int,
    memory_limit_bytes: int,
    codex_sidecar_events: list[dict[str, Any]] | None = None,
    refusal_events: list[dict[str, Any]] | None = None,
    agent_error: str | None = None,
) -> dict[str, Any]:
    run_summary = build_cli_run_summary(
        agent_type=agent_type,
        trajectory=trajectory,
        duration_seconds=duration,
        model_name=model_name,
        codex_sidecar_events=codex_sidecar_events,
        refusal_events=refusal_events,
        agent_error=agent_error,
    )
    metrics = run_summary.metrics
    metadata = {
        "duration_s": round(duration, 2),
        "model": model_name,
        "memory_limit_bytes": memory_limit_bytes,
        "run_summary": run_summary.model_dump(mode="json"),
    }

    if agent_type == "claudecode":
        claude_result = next(
            (event for event in reversed(trajectory) if event.get("type") == "result"),
            None,
        )
        if claude_result is not None:
            metadata["session_id"] = claude_result.get("session_id")
            usage = claude_result.get("usage")
            if isinstance(usage, dict):
                metadata["usage"] = usage
    elif agent_type == "openaicodex":
        thread_id = None
        for event in trajectory:
            if event.get("type") == "thread.started":
                thread_id = event.get("thread_id")
        if thread_id:
            metadata["thread_id"] = thread_id
    elif agent_type == "pi":
        session_id = None
        for event in trajectory:
            if event.get("type") == "session":
                session_id = event.get("id")
        if session_id:
            metadata["session_id"] = session_id
    elif agent_type == "grokbuild":
        session_id = None
        for event in trajectory:
            if event.get("sessionId"):
                session_id = event.get("sessionId")
        if session_id:
            metadata["session_id"] = session_id

    if metrics.total_cost_usd is not None:
        metadata["total_cost"] = metrics.total_cost_usd
    if metrics.turn_count is not None:
        metadata["n_turns"] = metrics.turn_count
    if metrics.step_count is not None:
        metadata["n_steps"] = metrics.step_count
    if "usage" not in metadata:
        canonical_usage = metrics.usage.model_dump(exclude_none=True)
        if canonical_usage:
            metadata["usage"] = canonical_usage

    metadata["timed_out"] = timed_out
    metadata["eval_timeout_seconds"] = eval_timeout
    metadata["oom_detected"] = oom_detected
    metadata["oom_restarts"] = oom_restarts
    if error_details:
        metadata["error_details"] = error_details

    return metadata
