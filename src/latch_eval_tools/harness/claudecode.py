import os
from pathlib import Path

from latch_eval_tools.harness._cli_runner import EVAL_TIMEOUT, _run_cli_agent
from latch_eval_tools.harness.utils import DEFAULT_DOCKER_IMAGE, load_data_instructions

CLAUDE_CODE_EXEC_NOTE = (
    "\n\nNote: Ending your turn ends this session. "
    "Nothing will resume or re-invoke you afterward."
)

MODEL_MAP = {
    "anthropic/claude-opus-4-6": "claude-opus-4-6",
    "anthropic/claude-opus-4-5": "claude-opus-4-5",
    "anthropic/claude-sonnet-4-6": "claude-sonnet-4-6",
    "anthropic/claude-sonnet-4-5": "claude-sonnet-4-5",
    "anthropic/claude-opus-4-7": "claude-opus-4-7",
    "anthropic/claude-sonnet-4-7": "claude-sonnet-4-7",
    "anthropic/claude-opus-4-8": "claude-opus-4-8",
}


def run_claudecode_task(
    task_prompt: str,
    work_dir: Path,
    model_name: str | None = None,
    eval_timeout: int = EVAL_TIMEOUT,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    memory_limit_bytes: int | None = None,
    system_prompt: str | None = None,
    prompt_suffix: str | None = load_data_instructions(),
    completion: bool = False,
    background_task_notification: bool = True,
) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required for Claude Code"
        )

    return _run_cli_agent(
        agent_type="claudecode",
        cli_command=["claude"],
        task_prompt=task_prompt,
        work_dir=work_dir,
        model_name=model_name,
        eval_timeout=eval_timeout,
        model_map=MODEL_MAP,
        docker_image=docker_image,
        memory_limit_bytes=memory_limit_bytes,
        system_prompt=system_prompt,
        prompt_suffix=prompt_suffix + CLAUDE_CODE_EXEC_NOTE
        if background_task_notification
        else prompt_suffix,
        completion=completion,
    )
