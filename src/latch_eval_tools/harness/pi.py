from pathlib import Path

from latch_eval_tools.harness._cli_runner import _run_cli_agent, EVAL_TIMEOUT
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
    )
