from datetime import datetime
from importlib.resources import files
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from latch_eval_tools.harness.utils import (
    DEFAULT_DOCKER_IMAGE,
    ensure_docker_image,
    get_agent_workspace_mount_args,
    get_agent_workspace_dir,
    get_memory_limit_bytes,
    is_docker_container_oom_killed,
    is_docker_container_running,
    load_trajectory_identifier,
    prompt_with_suffix,
    render_packaged_prompt,
)

EVAL_TIMEOUT = 600
ANTHROPIC_ENV_KEYS = {"ANTHROPIC_API_KEY"}
OPENAI_ENV_KEYS = {"OPENAI_API_KEY", "CODEX_API_KEY"}
PI_ENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "FIREWORKS_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "XAI_API_KEY",
}

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
}
AGENT_IDENTIFIER_KEYS = {
    "claudecode": "session_id",
    "openaicodex": "thread_id",
    "pi": "id",
}
PI_IGNORED_EVENT_TYPES = {"message_update", "tool_execution_update"}
PI_TOOL_TIMEOUT_EXTENSION_RELATIVE_PATH = Path(".latch_eval_tools", "tool_timeout.js")
PI_TOOL_TIMEOUT_EXTENSION_CONTAINER_PATH = (
    f"/workspace/{PI_TOOL_TIMEOUT_EXTENSION_RELATIVE_PATH}"
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
                "--settings",
                json.dumps({"showThinkingSummaries": True}),
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
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    if model_name and model_map:
        mapped_model = model_map.get(model_name, model_name)
        agent_cmd.extend(["--model", mapped_model])
    elif model_name:
        agent_cmd.extend(["--model", model_name])
    if agent_type != "pi" and resume_identifier is not None:
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
    container_state_mount = f"/root/{state_dir_name}"
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


def _append_codex_sidecar_reasoning(
    work_dir: Path,
    trajectory: list[dict],
) -> int:
    trajectory_file = work_dir / "trajectory.json"
    thread_id = load_trajectory_identifier(trajectory_file, "thread_id")
    if thread_id is None:
        return 0

    codex_dir = work_dir / AGENT_STATE_DIRS["openaicodex"]
    if not codex_dir.exists():
        return 0

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
    for source in sorted(codex_dir.rglob("*")):
        if source.is_dir() or source.is_symlink() or thread_id not in source.name:
            continue
        for line in source.read_text().splitlines():
            stripped = line.strip()
            if stripped == "":
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "response_item":
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
    agent_type: str,
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
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
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
    trajectory = []
    trajectory_file = work_dir / "trajectory.json"
    trajectory_file.write_text(json.dumps(trajectory, indent=2))
    eval_answer_file = agent_dir / "eval_answer.json"
    finished_file = agent_dir / "finished.txt"
    oom_detected = False
    oom_restarts = 0

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

                attempt_start_index = len(trajectory)
                agent_cmd = _build_agent_command(
                    agent_type=agent_type,
                    cli_command=cli_command,
                    model_name=model_name,
                    model_map=model_map,
                    claude_code_extra_args=claude_code_extra_args,
                    resume_identifier=resume_identifier,
                    system_prompt=system_prompt,
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
                                if (
                                    agent_type == "pi"
                                    and event["type"] in PI_IGNORED_EVENT_TYPES
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
                if timed_out_attempt:
                    log_file.write(
                        f"\n\nAgent timed out after {eval_timeout} seconds\n"
                    )
                    log_file.flush()
                    timed_out = True
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
                    if agent_type == "claudecode":
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
                                    AGENT_IDENTIFIER_KEYS["claudecode"],
                                )
                                if resume_identifier is not None:
                                    claudecode_answer_resumes += 1
                                    log_file.write(
                                        "\n\nClaude Code ended its turn without a "
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

    if agent_type == "openaicodex" and trajectory:
        appended_reasoning = _append_codex_sidecar_reasoning(work_dir, trajectory)
        if appended_reasoning > 0:
            print(
                "Appended "
                f"{appended_reasoning} Codex reasoning item(s) from sidecar to trajectory"
            )

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
    )

    return {"answer": agent_answer, "metadata": metadata}


def _extract_metadata(
    agent_type: str,
    trajectory: list[dict],
    duration: float,
    model_name: str | None,
    timed_out: bool,
    eval_timeout: int,
    error_details: dict | None,
    oom_detected: bool,
    oom_restarts: int,
    memory_limit_bytes: int,
) -> dict:
    metadata = {
        "duration_s": round(duration, 2),
        "model": model_name,
        "memory_limit_bytes": memory_limit_bytes,
    }

    if agent_type == "claudecode":
        claude_result = None
        for event in trajectory:
            if event.get("type") == "result":
                claude_result = event
                break
        if claude_result:
            metadata["total_cost"] = claude_result.get("total_cost_usd")
            metadata["n_turns"] = claude_result.get("num_turns")
            metadata["session_id"] = claude_result.get("session_id")
            metadata["usage"] = claude_result.get("usage")
    elif agent_type == "openaicodex":
        thread_id = None
        n_turns = 0
        total_usage = {"input_tokens": 0, "output_tokens": 0}

        for event in trajectory:
            event_type = event.get("type", "")
            if event_type == "thread.started":
                thread_id = event.get("thread_id")
            elif event_type == "turn.completed":
                n_turns += 1
                if "usage" in event:
                    usage = event["usage"]
                    total_usage["input_tokens"] += usage.get("input_tokens", 0)
                    total_usage["output_tokens"] += usage.get("output_tokens", 0)

        if thread_id:
            metadata["thread_id"] = thread_id
        if n_turns > 0:
            metadata["n_turns"] = n_turns
        if total_usage["input_tokens"] > 0 or total_usage["output_tokens"] > 0:
            metadata["usage"] = total_usage
    elif agent_type == "pi":
        session_id = None
        n_turns = 0
        total_cost = 0
        total_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

        for event in trajectory:
            if event.get("type") == "session":
                session_id = event.get("id")
            elif event.get("type") == "turn_end":
                n_turns += 1
            elif event.get("type") == "message_end":
                message = event.get("message")
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                usage = message["usage"]
                total_usage["input_tokens"] += usage["input"]
                total_usage["output_tokens"] += usage["output"]
                total_usage["cache_read_tokens"] += usage["cacheRead"]
                total_usage["cache_write_tokens"] += usage["cacheWrite"]
                total_cost += usage["cost"]["total"]

        if session_id:
            metadata["session_id"] = session_id
        if n_turns > 0:
            metadata["n_turns"] = n_turns
        if any(total_usage.values()):
            metadata["usage"] = total_usage
        if total_cost > 0:
            metadata["total_cost"] = total_cost

    metadata["timed_out"] = timed_out
    metadata["eval_timeout_seconds"] = eval_timeout
    metadata["oom_detected"] = oom_detected
    metadata["oom_restarts"] = oom_restarts
    if error_details:
        metadata["error_details"] = error_details

    return metadata
