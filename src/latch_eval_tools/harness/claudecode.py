import os
from pathlib import Path
from latch_eval_tools.harness._cli_runner import _run_cli_agent, EVAL_TIMEOUT

MODEL_MAP = {
    "anthropic/claude-opus-4-6": "claude-opus-4-6",
    "anthropic/claude-opus-4-5": "claude-opus-4-5",
    "anthropic/claude-sonnet-4-5": "claude-sonnet-4-5",
}


def run_claudecode_task(
    task_prompt: str,
    work_dir: Path,
    model_name: str | None = None,
    eval_timeout: int = EVAL_TIMEOUT,
) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable is required for Claude Code")

    return _run_cli_agent(
        agent_type="claudecode",
        cli_command=["claude"],
        task_prompt=task_prompt,
        work_dir=work_dir,
        model_name=model_name,
        eval_timeout=eval_timeout,
        model_map=MODEL_MAP,
    )
