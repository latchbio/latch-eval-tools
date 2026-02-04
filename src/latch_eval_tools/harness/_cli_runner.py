import json
import os
import pwd
import re
import shutil
import stat
import subprocess
import time
from pathlib import Path

EVAL_TIMEOUT = 600


def _run_cli_agent(
    agent_type: str,
    cli_command: list[str],
    task_prompt: str,
    work_dir: Path,
    model_name: str | None = None,
    eval_timeout: int = EVAL_TIMEOUT,
    model_map: dict[str, str] | None = None,
) -> dict:
    try:
        subprocess.run(
            cli_command + ["--version"],
            capture_output=True,
            check=True,
            timeout=5
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        cli_name = " ".join(cli_command)
        raise FileNotFoundError(f"{cli_name} CLI not found. Please install it first.")

    agent_log_file = work_dir / "agent_output.log"
    if agent_log_file.exists():
        agent_log_file.unlink()

    enhanced_prompt = _enhance_prompt_with_local_files(task_prompt, work_dir)
    enhanced_prompt += f"""

CRITICAL: You must write eval_answer.json BEFORE signaling completion in the {work_dir}.
Correct order: 1) Perform analysis 2) Write eval_answer.json with your answer 3) Exit"""

    if agent_type == "claudecode":
        cmd = cli_command + ["--print", "--dangerously-skip-permissions", "--verbose", "--output-format", "stream-json"]
    elif agent_type == "openaicodex":
        cmd = cli_command + ["--full-auto", "--skip-git-repo-check", "--json"]
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    if model_name and model_map:
        mapped_model = model_map.get(model_name, model_name)
        cmd.extend(["--model", mapped_model])
    elif model_name:
        cmd.extend(["--model", model_name])

    run_as_claude_user = agent_type == "claudecode" and os.geteuid() == 0
    if run_as_claude_user:
        try:
            pwd.getpwnam("claude")
            home_dir = Path.home()
            current_mode = home_dir.stat().st_mode
            home_dir.chmod(current_mode | stat.S_IXOTH)

            eval_cache_dir = home_dir / ".eval_cache"
            if eval_cache_dir.exists():
                shutil.chown(eval_cache_dir, user="claude", group="claude")
                for item in eval_cache_dir.rglob("*"):
                    try:
                        shutil.chown(item, user="claude", group="claude")
                    except PermissionError:
                        pass

            work_dir.chmod(0o777)
        except KeyError:
            run_as_claude_user = False

    env = os.environ.copy()
    
    if agent_type == "openaicodex":
        if "CODEX_API_KEY" not in env and "OPENAI_API_KEY" in env:
            env["CODEX_API_KEY"] = env["OPENAI_API_KEY"]

    start_time = time.time()
    timed_out = False
    trajectory = []

    try:
        if run_as_claude_user:
            env_vars = [f"{k}={v}" for k, v in env.items() if k.endswith("_API_KEY")]
            cmd = ["runuser", "-u", "claude", "--", "env"] + env_vars + cmd

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(work_dir),
            env=env,
            text=True,
        )

        try:
            stdout, stderr = process.communicate(
                input=enhanced_prompt,
                timeout=eval_timeout
            )

            with open(agent_log_file, 'w') as log_file:
                log_file.write(stdout)
                if stderr:
                    log_file.write(f"\n\nSTDERR:\n{stderr}")

            for line in stdout.strip().split('\n'):
                if line:
                    try:
                        event = json.loads(line)
                        trajectory.append(event)
                    except json.JSONDecodeError:
                        print(f"Warning: Failed to parse JSON: {line}")

        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            stdout, stderr = process.communicate()
            with open(agent_log_file, 'w') as log_file:
                log_file.write(stdout)
                log_file.write(f"\n\nAgent timed out after {eval_timeout} seconds")

    except Exception as e:
        with open(agent_log_file, 'a') as f:
            f.write(f"\nError running {agent_type}: {e}")

    duration = time.time() - start_time
    print(f"Agent output saved to: {agent_log_file}")

    if trajectory:
        trajectory_file = work_dir / "trajectory.json"
        trajectory_file.write_text(json.dumps(trajectory, indent=2))
        print(f"Trajectory saved to: {trajectory_file}")

    eval_answer_file = work_dir / "eval_answer.json"
    agent_answer = None
    error_details = None

    if not eval_answer_file.exists():
        log_tail = ""
        if agent_log_file.exists():
            log_content = agent_log_file.read_text()
            log_tail = log_content[-1000:]

        error_msg = "Agent timed out" if timed_out else "Agent did not create eval_answer.json"
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


def _enhance_prompt_with_local_files(task_prompt: str, work_dir: Path) -> str:
    contextual_data_match = re.search(
        r'<ContextualNodeData>(.*?)</ContextualNodeData>',
        task_prompt,
        re.DOTALL
    )

    if not contextual_data_match:
        return task_prompt

    try:
        contextual_data = json.loads(contextual_data_match.group(1))
    except json.JSONDecodeError:
        return task_prompt

    local_files = []
    for item in contextual_data:
        if 'local_path' in item:
            local_files.append(item['local_path'])

    if not local_files:
        return task_prompt

    file_list = "\n".join([f"- {f}" for f in local_files])
    enhancement = f"\n\nThe following data files are available in your current working directory:\n{file_list}\n\nUse these local filenames to access the data.\n"

    parts = task_prompt.split('<ContextualNodeData>')
    if len(parts) == 2:
        return parts[0] + enhancement + '<ContextualNodeData>' + parts[1]

    return task_prompt
