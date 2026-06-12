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
    load_data_instructions,
    load_trajectory_identifier,
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
PI_TOOL_TIMEOUT_EXTENSION_RELATIVE_PATH = Path(
    ".latch_eval_tools", "tool_timeout.js"
)
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
    

def _apply_extra_args(cmd: list[str], extra_args: list[str]) -> list[str]:
    cmd = list(cmd)
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        # Check if the current flag has a corresponding value
        has_value = i + 1 < len(extra_args) and not extra_args[i + 1].startswith("-")
        if has_value:
            value = extra_args[i + 1]
            if arg == "-c":
                # match by config key, not just flag
                config_key = value.split("=")[0]
                replaced = False
                for j, cmd_arg in enumerate(cmd):
                    if cmd_arg == "-c" and j + 1 < len(cmd) and cmd[j + 1].split("=")[0] == config_key:
                        cmd[j + 1] = value
                        replaced = True
                        break
                if not replaced:
                    cmd.extend(["-c", value])
            elif arg in cmd:
                cmd[cmd.index(arg) + 1] = value
            else:
                cmd.extend([arg, value])
            i += 2
        else:
            if arg not in cmd:
                cmd.append(arg)
            i += 1
    return cmd 


def _build_agent_command(
    agent_type: str,
    cli_command: list[str],
    model_name: str | None,
    model_map: dict[str, str] | None,
    extra_args: list[str] | None = None,
    resume_identifier: str | None = None,
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
                "--tools",
                "Bash",
            ]
        )
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
            ]
        )
    elif agent_type == "pi":
        agent_cmd = list(cli_command)
        agent_cmd.extend(["--mode", "json", "--print"])
        if resume_identifier is not None:
            agent_cmd.extend(["--session", resume_identifier])
        agent_cmd.extend(["--thinking", "xhigh"])
        agent_cmd.extend(["--extension", PI_TOOL_TIMEOUT_EXTENSION_CONTAINER_PATH])
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    if model_name and model_map:
        mapped_model = model_map.get(model_name, model_name)
        agent_cmd.extend(["--model", mapped_model])
    elif model_name:
        agent_cmd.extend(["--model", model_name])
    if agent_type != "pi" and resume_identifier is not None:
        agent_cmd.append(resume_identifier)

    if extra_args:
        agent_cmd = _apply_extra_args(agent_cmd, extra_args)

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
        assert result.returncode == 0,f"Failed to start container {container_name}"
    except Exception as e:
        print(f"Error starting container {container_name}: {e}")
        raise e


def _run_cli_agent(
    agent_type: str,
    cli_command: list[str],
    task_prompt: str,
    work_dir: Path,
    model_name: str | None = None,
    eval_timeout: int = EVAL_TIMEOUT,
    model_map: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    memory_limit_bytes: int | None = None,
) -> dict:
    agent_log_file = work_dir / "agent_output.log"
    if agent_log_file.exists():
        agent_log_file.unlink()
    # todo(tim): clean up instructions based on early exit
    enhanced_prompt = f"{task_prompt}\n{load_data_instructions()}"

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
    if agent_type == "pi":
        env_flags.extend(
            [
                "-e",
                "PI_SKIP_VERSION_CHECK=1",
                "-e",
                "PI_TELEMETRY=0",
            ]
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

            while True:
                remaining_timeout = deadline - time.time()
                if remaining_timeout <= 0:
                    timed_out = True
                    log_file.write(
                        f"\n\nAgent timed out after {eval_timeout} seconds\n"
                    )
                    log_file.flush()
                    break

                agent_cmd = _build_agent_command(
                    agent_type=agent_type,
                    cli_command=cli_command,
                    model_name=model_name,
                    model_map=model_map,
                    extra_args=extra_args,
                    resume_identifier=resume_identifier,
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
                            if agent_type == "pi" and eval_answer_file.exists():
                                json.loads(eval_answer_file.read_text())
                                answer_submitted = True
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

    if trajectory:
        persist_trajectory()
        print(f"Trajectory saved to: {trajectory_file}")

    eval_answer_file = agent_dir / "eval_answer.json"
    agent_answer = None
    error_details = None

    if not eval_answer_file.exists():
        log_tail = ""
        if agent_log_file.exists():
            log_content = agent_log_file.read_text()
            log_tail = log_content[-1000:]

        if timed_out:
            error_msg = "Agent timed out"
        elif agent_error is not None:
            error_msg = f"{type(agent_error).__name__}: {agent_error}"
        else:
            error_msg = "Agent did not create eval_answer.json"
        error_details = {
            "error": error_msg,
            "timed_out": timed_out,
            "log_tail": log_tail,
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
        agent_cmd=agent_cmd
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
    agent_cmd: list[str] | None = None
) -> dict:
    metadata = {
        "duration_s": round(duration, 2),
        "model": model_name,
        "memory_limit_bytes": memory_limit_bytes,
        "agent_cmd": agent_cmd
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
