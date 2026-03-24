from datetime import datetime
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
    load_data_instructions,
    preload_cached_docker_image,
    resolve_data_mounts,
)

EVAL_TIMEOUT = 600
DOCKER_ENV_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CODEX_API_KEY")


def copy_and_teardown(container_name: str, agent_type: str, work_dir: Path) -> None:
    source_by_agent = {
        "claudecode": "/root/.claude",
        "openaicodex": "/root/.codex",
    }
    source = source_by_agent.get(agent_type)

    try:
        if source is None:
            return

        container_dest = f"/workspace/{Path(source).name}"
        host_dest = work_dir / Path(source).name
        copy_result = subprocess.run(
            ["docker", "exec", container_name, "cp", "-r", source, container_dest],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if copy_result.returncode == 0:
            print(f"Copied harness state from {source} to {host_dest}")
            return

        stderr = copy_result.stderr.strip()
        if stderr:
            print(f"Error copying harness state from {source}: {stderr}")
        else:
            print(f"cp failed (rc={copy_result.returncode}) for {source} in {container_name}")
    except Exception as exc:
        print(f"Failed to copy harness state from {container_name}:{source}: {exc}")
    finally:
        try:
            remove_result = subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if remove_result.returncode != 0:
                stderr = remove_result.stderr.strip()
                if stderr and "No such container" not in stderr:
                    print(f"Failed to remove container {container_name}: {stderr}")
        except Exception as exc:
            print(f"Failed to remove container {container_name}: {exc}")


def _run_cli_agent(
    agent_type: str,
    cli_command: list[str],
    task_prompt: str,
    work_dir: Path,
    model_name: str | None = None,
    eval_timeout: int = EVAL_TIMEOUT,
    model_map: dict[str, str] | None = None,
    claude_code_extra_args: list[str] | None = ["--tools", "Bash"],
    docker_image: str = DEFAULT_DOCKER_IMAGE,
) -> dict:
    agent_log_file = work_dir / "agent_output.log"
    if agent_log_file.exists():
        agent_log_file.unlink()

    enhanced_prompt = f"{task_prompt}\n{load_data_instructions()}"


    if agent_type == "claudecode":
        agent_cmd = cli_command + ["--print", "--dangerously-skip-permissions", "--verbose", "--output-format", "stream-json"] + claude_code_extra_args if claude_code_extra_args else []
    elif agent_type == "openaicodex":
        agent_cmd = cli_command + ["--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--json", "-c", 'model_reasoning_effort="xhigh"']
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    if model_name and model_map:
        mapped_model = model_map.get(model_name, model_name)
        agent_cmd.extend(["--model", mapped_model])
    elif model_name:
        agent_cmd.extend(["--model", model_name])

    env = os.environ.copy()

    if agent_type == "openaicodex":
        if "CODEX_API_KEY" not in env and "OPENAI_API_KEY" in env:
            env["CODEX_API_KEY"] = env["OPENAI_API_KEY"]

    if not docker_image:
        raise ValueError("docker_image is required for CLI harnesses")

    preload_cached_docker_image()
    ensure_docker_image(docker_image)
    data_mounts = resolve_data_mounts(work_dir)
    env_flags: list[str] = []
    for key in DOCKER_ENV_KEYS:
        value = env.get(key)
        if value:
            env_flags.extend(["-e", f"{key}={value}"])
    container_name = f"eval-{agent_type}-{uuid.uuid4().hex[:8]}"

    agent_start_time = time.time()
    agent_finished_at = agent_start_time
    timed_out = False
    agent_error: Exception | None = None
    trajectory = []
    trajectory_file = work_dir / "trajectory.json"
    trajectory_file.write_text(json.dumps(trajectory, indent=2))

    trajectory_lock = threading.Lock()

    def persist_trajectory():
        with trajectory_lock:
            trajectory_file.write_text(json.dumps(trajectory, indent=2))

    try:
        subprocess.run(
            [
                "docker", "create",
                "--name", container_name,
                "-i",
                "-v", f"{work_dir}:/workspace",
                "-w", "/workspace",
                *data_mounts,
                *env_flags,
                docker_image,
                "sleep", "infinity",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["docker", "start", container_name],
            check=True,
            capture_output=True,
            text=True,
        )

        with open(agent_log_file, "w") as log_file:
            process = subprocess.Popen(
                ["docker", "exec", "-i", container_name, *agent_cmd],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(work_dir),
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
                        log_file.write(line)
                        log_file.flush()

                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            event = json.loads(stripped)
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
                process.stdin.write(enhanced_prompt)
                process.stdin.close()

            agent_start_time = time.time()

            try:
                process.wait(timeout=eval_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait()
                log_file.write(f"\n\nAgent timed out after {eval_timeout} seconds\n")
                log_file.flush()

            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

    except Exception as e:
        agent_error = e
        with open(agent_log_file, 'a') as f:
            f.write(f"\nError running {agent_type}: {e}")
    finally:
        agent_finished_at = time.time()
        copy_and_teardown(container_name, agent_type, work_dir)

    duration = agent_finished_at - agent_start_time
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Agent output saved to: {agent_log_file}")

    if trajectory:
        persist_trajectory()
        print(f"Trajectory saved to: {trajectory_file}")

    eval_answer_file = work_dir / "eval_answer.json"
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
            "log_tail": log_tail
        }
        print(f"\nWarning: {error_msg}")
    else:
        try:
            agent_answer = json.loads(eval_answer_file.read_text())
        except json.JSONDecodeError as e:
            error_details = {
                "error": f"Failed to parse eval_answer.json: {e}",
                "file_contents": eval_answer_file.read_text()[:500]
            }
            print(f"\nWarning: Failed to parse eval_answer.json: {e}")

    metadata = _extract_metadata(agent_type, trajectory, duration, model_name, timed_out, eval_timeout, error_details)

    return {"answer": agent_answer, "metadata": metadata}


def _extract_metadata(
    agent_type: str,
    trajectory: list[dict],
    duration: float,
    model_name: str | None,
    timed_out: bool,
    eval_timeout: int,
    error_details: dict | None,
) -> dict:
    metadata = {
        "duration_s": round(duration, 2),
        "model": model_name,
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

    if timed_out:
        metadata["timed_out"] = True
        metadata["eval_timeout_seconds"] = eval_timeout
    if error_details:
        metadata["error_details"] = error_details

    return metadata


