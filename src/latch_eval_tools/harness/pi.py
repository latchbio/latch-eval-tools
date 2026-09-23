from pathlib import Path

from latch_eval_tools.harness._cli_runner import (
    EVAL_TIMEOUT,
    CliChunkResult,
    _run_cli_agent,
    _run_cli_chunk,
)
from latch_eval_tools.harness.utils import DEFAULT_DOCKER_IMAGE, load_data_instructions


def _map_model_name(model_name: str | None) -> str | None:
    # Pi uses google/..., existing eval harness calls use gemini/...
    if model_name is not None and model_name.startswith("gemini/"):
        return f"google/{model_name.removeprefix('gemini/')}"
    return model_name


def run_pi_task(
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
    operation_timeout: int = 0,
    completion_file_path: str | None = None,
) -> dict:
    return _run_cli_agent(
        agent_type="pi",
        cli_command=["pi"],
        task_prompt=task_prompt,
        work_dir=work_dir,
        model_name=_map_model_name(model_name),
        eval_timeout=eval_timeout,
        docker_image=docker_image,
        memory_limit_bytes=memory_limit_bytes,
        system_prompt=system_prompt,
        prompt_suffix=prompt_suffix,
        completion=completion,
        benchmark=benchmark,
        operation_timeout=operation_timeout,
        completion_file_path=completion_file_path,
    )


def run_pi_chunk(
    container_name: str,
    prompt: str,
    work_dir: Path,
    max_turns: int,
    model_name: str | None = None,
    system_prompt: str | None = None,
    resume_identifier: str | None = None,
    fork: bool = False,
    timeout: int = EVAL_TIMEOUT,
) -> CliChunkResult:
    return _run_cli_chunk(
        agent_type="pi",
        cli_command=["pi"],
        container_name=container_name,
        prompt=prompt,
        work_dir=work_dir,
        max_turns=max_turns,
        model_name=_map_model_name(model_name),
        system_prompt=system_prompt,
        resume_identifier=resume_identifier,
        fork=fork,
        timeout=timeout,
    )
