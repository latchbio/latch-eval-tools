import io
from datetime import datetime
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
import time
from typing import Any
import yaml

from latch_eval_tools.harness.utils import (
    DEFAULT_DOCKER_IMAGE,
    ensure_docker_image,
    get_memory_limit_bytes,
    is_docker_container_oom_killed,
    is_docker_container_running,
    load_data_instructions,
    read_packaged_prompt,
    render_packaged_prompt,
    resolve_data_mounts,
)

OPERATION_TIMEOUT = 300
EVAL_TIMEOUT = 600
OOM_EXIT_CODE = 137
MAX_OOM_RESTARTS = 10

class AgentTimeoutError(KeyboardInterrupt):
    # Use a KeyboardInterrupt-style base so model/provider retry layers that catch
    # Exception do not swallow the eval-level timeout and keep running past deadline.
    pass


def _timeout_handler(signum, frame):
    raise AgentTimeoutError("Agent exceeded time limit")


class StreamingLogFile: 
    def __init__(self, file_path):
        self.file_path = file_path
        self.buffer = io.StringIO()

    def write(self, data):
        self.buffer.write(data)
        with open(self.file_path, 'a') as f:
            f.write(data)
            f.flush()

    def flush(self):
        pass

    def getvalue(self):
        return self.buffer.getvalue()


def _persist_agent_trajectory(agent, trajectory_file: Path):
    trajectory_data = agent.serialize()
    temp_file = trajectory_file.with_suffix(".tmp")
    temp_file.write_text(json.dumps(trajectory_data, indent=2))
    temp_file.replace(trajectory_file)


def _render_logged_message_content(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        rendered_content = content
    elif content is None:
        rendered_content = ""
    else:
        rendered_content = json.dumps(content, indent=2)

    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return rendered_content

    rendered_tool_calls = []
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        rendered_tool_calls.append(
            f"{function.get('name', 'unknown')}({function.get('arguments', '')})"
        )

    tool_call_summary = "Tool calls:\n" + "\n".join(rendered_tool_calls)
    if rendered_content:
        return f"{rendered_content}\n{tool_call_summary}"
    return tool_call_summary


def _patch_agent_for_progress(log_file, trajectory_file: Path, agent_class):
    if getattr(agent_class, "_latch_progress_patch_applied", False):
        return

    original_add_messages = agent_class.add_messages

    def patched_add_messages(self, *messages):
        added_messages = original_add_messages(self, *messages)

        with open(log_file, "a") as f:
            for message in added_messages:
                role = message.get("role")
                content = _render_logged_message_content(message)
                if role == "assistant":
                    step_num = len([m for m in self.messages if m.get("role") == "assistant"])
                    f.write(f"\n[Step {step_num}]\n")
                    f.write(f"Assistant: {content}\n")
                elif role in {"tool", "user"} and len(self.messages) > 2:
                    f.write(f"Observation: {content}\n")
                elif role == "exit":
                    f.write(f"\n[Exit]\n{content}\n")
            f.flush()

        _persist_agent_trajectory(self, trajectory_file)
        return added_messages

    agent_class.add_messages = patched_add_messages
    agent_class._latch_progress_patch_applied = True


def get_model_kwargs(model_name: str) -> dict[str, Any]:
    if model_name in {"openai/gpt-5.4", "openai/gpt-5.3-codex", "openai/gpt-5.3", "openai/gpt-5.2"}:
        return {"model_kwargs": {"reasoning": {"effort": "xhigh"}},"model_class":"litellm_response"}
    elif model_name in {"openai/gpt-5.1"}:
        return {"model_kwargs": {"reasoning": {"effort": "high"}},"model_class":"litellm_response"}
    elif model_name in {"anthropic/claude-opus-4-6"}:
        return {"model_kwargs": {"thinking": {"type": "adaptive"}}}
    elif model_name.startswith("anthropic/"):
        return {"model_kwargs": {"thinking": {"type": "enabled", "budget_tokens": 32000}}}
    elif model_name.startswith("gemini/"):
        return {"model_kwargs": {"generationConfig": {"thinkingConfig": {"thinkingLevel":"HIGH"}}}}
    elif model_name.startswith("xai/") and model_name.endswith("-reasoning"):
        return {"model_class":"litellm_response"}
    elif model_name == "openai/moonshotai/Kimi-K2.6":
        return {
            "cost_tracking": "ignore_errors",
            "model_kwargs": {
                "api_base": "https://inference.baseten.co/v1",
                "api_key": os.environ["BASETEN_API_KEY"],
                "extra_body": {"thinking": {"type": "enabled","keep":"all"}},
            },
        }
    else:
        return {}
    

def run_minisweagent_task(
    task_prompt: str,
    work_dir: Path,
    model_name: str,
    agent_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    env_config: dict[str, Any] | None = None,
    operation_timeout: int = OPERATION_TIMEOUT,
    eval_timeout: int = EVAL_TIMEOUT,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    memory_limit_bytes: int | None = None,
) -> dict:
    """Run MiniSWE agent on a task.
    
    Args:
        task_prompt: Task description for the agent
        work_dir: Working directory for the agent
        model_name: Optional model name (e.g., "anthropic/claude-sonnet-4")
        agent_config: Optional agent configuration dict
        operation_timeout: Timeout for individual operations (seconds)
        eval_timeout: Timeout for entire evaluation (seconds)
    
    Returns:
        dict with keys "answer" (parsed JSON or None) and "metadata"
    """
    assert model_name is not None, f"Expect a model name got {model_name}"
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.exceptions import Submitted
    from minisweagent.models import get_model
    from minisweagent.exceptions import LimitsExceeded


    class FlexibleAgent(DefaultAgent):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._override_start_time = time.monotonic()

        def step(self) -> list[dict]:
            if time.monotonic() - self._override_start_time > eval_timeout:
                raise LimitsExceeded({
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                })
            return super().step()

    class FlexibleDockerEnvironment(DockerEnvironment):
        completion_marker = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

        def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
            nonlocal oom_detected, oom_restarts, container_restarts
            output = super().execute(action, cwd=cwd, timeout=timeout)

            if output.get("returncode") == 0 and not output.get("exception_info"):
                return output

            container_running = bool(self.container_id) and is_docker_container_running(
                self.container_id,
                docker_executable=self.config.executable,
            )
            container_oom_killed = bool(self.container_id) and is_docker_container_oom_killed(
                self.container_id,
                docker_executable=self.config.executable,
            )
            command_hit_oom = (
                output.get("returncode") == OOM_EXIT_CODE or container_oom_killed
            )
            if command_hit_oom:
                oom_detected = True
                if oom_restarts >= MAX_OOM_RESTARTS:
                    raise LimitsExceeded({
                        "role": "exit",
                        "content": "LimitsExceeded",
                        "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                    })
                oom_restarts += 1

            if not container_running:
                self.cleanup()
                self.container_id = None
                self._start_container()
                container_restarts += 1
                failure_reason = (
                    "after likely exceeding the execution container memory limit."
                    if command_hit_oom
                    else "because the execution container stopped unexpectedly."
                )
                container_action = "A fresh execution container has been started."
            elif command_hit_oom:
                failure_reason = (
                    "after likely exceeding the execution container memory limit."
                )
                container_action = (
                    "The existing execution container is still running, so only "
                    "that command needs to be retried."
                )
            else:
                return output

            recovery_message = render_packaged_prompt(
                "miniswe_memory_warning.md",
                failure_reason=failure_reason,
                container_action=container_action,
                workspace_dir=self.config.cwd,
            )
            existing_output = output.get("output", "")
            if existing_output and not existing_output.endswith("\n"):
                existing_output += "\n"
            output["output"] = f"{existing_output}{recovery_message}"
            output.setdefault("extra", {})
            output["extra"]["container_restarted"] = not container_running
            output["extra"]["oom_detected"] = command_hit_oom
            output["extra"]["oom_restarts"] = oom_restarts
            return output

        def _check_finished(self, output: dict):
            """Raises Submitted if the output indicates task completion."""
            lines = output.get("output", "").lstrip().splitlines(keepends=True)
            if lines and lines[0].strip() == self.completion_marker and output["returncode"] == 0:
                submission = "".join(lines[1:])
                if not (work_dir / "eval_answer.json").exists():
                    return
                raise Submitted(
                    {
                        "role": "exit",
                        "content": submission,
                        "extra": {"exit_status": "Submitted", "submission": submission},
                    }
                )


    original_dir = os.getcwd()

    agent_log_file = work_dir / "agent_output.log"
    trajectory_file = work_dir / "trajectory.json"
    trajectory_file.write_text(json.dumps({"messages": []}, indent=2))
    _patch_agent_for_progress(agent_log_file, trajectory_file, FlexibleAgent)
    if agent_log_file.exists():
        agent_log_file.unlink()

    captured_output = StreamingLogFile(agent_log_file)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    class TeeOutput:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                if hasattr(stream, 'flush'):
                    stream.flush()

        def flush(self):
            for stream in self.streams:
                if hasattr(stream, 'flush'):
                    stream.flush()

    agent = None
    timed_out = False
    agent_error: Exception | None = None
    oom_detected = False
    oom_restarts = 0
    container_restarts = 0
    try:
        os.chdir(str(work_dir))



        enhanced_prompt = task_prompt

        enhanced_prompt = f"{task_prompt}\n{load_data_instructions()}"
        config = yaml.safe_load(read_packaged_prompt("miniswe_config.yaml"))
        effective_agent_config: dict[str, Any] = config["agent"] | (agent_config if isinstance(agent_config, dict) else {})
        effective_env_config: dict[str, Any] = config["environment"] | (env_config if isinstance(env_config, dict) else {})
        effective_model_config: dict[str, Any] = (
            config["model"] | get_model_kwargs(model_name) | (model_config if isinstance(model_config, dict) else {})
        )
        if not docker_image:
            raise ValueError("docker_image is required for mini-swe Docker execution")
        if memory_limit_bytes is None:
            memory_limit_bytes = get_memory_limit_bytes()
        data_mounts = resolve_data_mounts(work_dir)
        docker_env_config = effective_env_config | {
            "image": docker_image,
            "cwd": "/workspace",
            "run_args": [
                "--rm",
                "--memory",
                str(memory_limit_bytes),
                "--memory-swap",
                str(memory_limit_bytes),
                "-v",
                f"{work_dir}:/workspace",
                *data_mounts,
            ],
            "timeout": operation_timeout,
            "container_timeout": str(eval_timeout),
        }

        if model_name is not None and model_name.startswith("mistral/"):
            os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

        ensure_docker_image(docker_image)

        sys.stdout = TeeOutput(original_stdout, captured_output)
        sys.stderr = TeeOutput(original_stderr, captured_output)
        model = get_model(model_name, config=effective_model_config)
        try:
            env = FlexibleDockerEnvironment(**docker_env_config)
        except subprocess.CalledProcessError as e:
            stdout = (e.stdout or "").strip()
            stderr = (e.stderr or "").strip()
            raise RuntimeError(
                "Failed to start mini-swe Docker environment.\n"
                f"Command: {shlex.join(e.cmd)}\n"
                f"stdout:\n{stdout or '<empty>'}\n"
                f"stderr:\n{stderr or '<empty>'}"
            ) from e
        agent = FlexibleAgent(model, env, **effective_agent_config)

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(eval_timeout)

        try:
            agent.run(enhanced_prompt)
        except AgentTimeoutError:
            timed_out = True
            print(f"\nAgent timed out after {eval_timeout} seconds")
        except Submitted:
            pass
        except Exception as e:
            agent_error = e
            import traceback
            traceback.print_exc()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

            sys.stdout = original_stdout
            sys.stderr = original_stderr

            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Agent output saved to: {agent_log_file}")

            if hasattr(agent, "messages"):
                _persist_agent_trajectory(agent, trajectory_file)
                print(f"Agent trajectory saved to: {trajectory_file}")
                print(f"  Total message exchanges: {len(agent.messages)}")

        eval_answer_file = work_dir / "eval_answer.json"
        agent_answer = None
        error_details = None

        if not eval_answer_file.exists():
            agent_log_file = work_dir / "agent_output.log"
            log_tail = ""
            if agent_log_file.exists():
                log_content = agent_log_file.read_text()
                log_tail = log_content[-1000:]

            trajectory_info = f"Agent had {len(agent.messages)} message exchanges."

            if timed_out:
                error_msg = "Agent timed out"
            elif agent_error is not None:
                error_msg = f"{type(agent_error).__name__}: {agent_error}"
            else:
                error_msg = "Agent did not create eval_answer.json"
            error_details = {
                "error": error_msg,
                "timed_out": timed_out,
                "trajectory_info": trajectory_info,
                "log_tail": log_tail
            }
            print(f"\nWarning: {error_msg}. {trajectory_info}")
        else:
            try:
                agent_answer = json.loads(eval_answer_file.read_text())
            except json.JSONDecodeError as e:
                error_details = {
                    "error": f"Failed to parse eval_answer.json: {e}",
                    "file_contents": eval_answer_file.read_text()[:500]
                }
                print(f"\nWarning: Failed to parse eval_answer.json: {e}")

        metadata = {}
        if agent is not None:
            metadata["total_cost"] = agent.cost
            metadata["n_steps"] = agent.n_calls
            metadata["n_messages"] = len(agent.messages)
        metadata["memory_limit_bytes"] = memory_limit_bytes
        metadata["oom_restarts"] = oom_restarts
        metadata["container_restarts"] = container_restarts
        if timed_out:
            metadata["timed_out"] = True
            metadata["eval_timeout_seconds"] = eval_timeout
        if oom_detected:
            metadata["oom_detected"] = True
        if error_details:
            metadata["error_details"] = error_details

        metadata["harness_config"] = {
            "agent_config": effective_agent_config,
            "model_config": effective_model_config,
            "env_config": docker_env_config,
        }

        return {"answer": agent_answer, "metadata": metadata}

    finally:
        os.chdir(original_dir)



