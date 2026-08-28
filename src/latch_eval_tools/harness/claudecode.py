import json
import os
from pathlib import Path

from latch_eval_tools.harness._cli_runner import EVAL_TIMEOUT, _run_cli_agent
from latch_eval_tools.harness.utils import DEFAULT_DOCKER_IMAGE, load_data_instructions

MODEL_MAP = {
    "anthropic/claude-opus-4-6": "claude-opus-4-6",
    "anthropic/claude-opus-4-5": "claude-opus-4-5",
    "anthropic/claude-sonnet-4-6": "claude-sonnet-4-6",
    "anthropic/claude-sonnet-4-5": "claude-sonnet-4-5",
    "anthropic/claude-opus-4-7": "claude-opus-4-7",
    "anthropic/claude-sonnet-4-7": "claude-sonnet-4-7",
    "anthropic/claude-opus-4-8": "claude-opus-4-8",
    "anthropic/claude-halva-eap": "claude-halva-eap",
    "anthropic/claude-fable-5": "claude-fable-5",
    "anthropic/claude-melon-lp-eap": "claude-melon-lp-eap",
}


def _switch_models_on_flag_args(value: bool | None) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError("switch_models_on_flag must be a bool or None")
    settings = json.dumps(
        {"switchModelsOnFlag": value},
        separators=(",", ":"),
    )
    return ["--settings", settings]


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
    benchmark: bool = False,
    switch_models_on_flag: bool | None = None,
) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required for Claude Code"
        )

    if prompt_suffix is None:
        prompt_suffix = ""

    return _run_cli_agent(
        agent_type="claudecode",
        cli_command=["claude"],
        task_prompt=task_prompt,
        work_dir=work_dir,
        model_name=model_name,
        eval_timeout=eval_timeout,
        model_map=MODEL_MAP,
        claude_code_extra_args=_switch_models_on_flag_args(switch_models_on_flag),
        docker_image=docker_image,
        memory_limit_bytes=memory_limit_bytes,
        system_prompt=system_prompt,
        prompt_suffix=prompt_suffix
        + "\n\nNote: Do not end your turn while a background process is still"
        " running. Nothing will alert you it is done. Ending your turn may"
        " complete the session and kill the process; instead, block"
        " synchronously and poll until the job finishes before returning.",
        completion=completion,
        benchmark=benchmark,
    )
