import os
from pathlib import Path

from latch_eval_tools.harness._cli_runner import EVAL_TIMEOUT, _run_cli_agent
from latch_eval_tools.harness.utils import DEFAULT_DOCKER_IMAGE, load_data_instructions

MODEL_MAP = {
    "xai/grok-4.6": "grok-4.6",
    "xai/grok-4.3": "grok-4.3",
    "xai/grok-4": "grok-4",
    "xai/grok-code-fast-1": "grok-code-fast-1",
}


def run_grokbuild_task(
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
) -> dict:
    if not os.environ.get("XAI_API_KEY"):
        raise ValueError(
            "XAI_API_KEY environment variable is required for grok-build"
        )

    if prompt_suffix is None:
        prompt_suffix = ""

    return _run_cli_agent(
        agent_type="grokbuild",
        cli_command=["grok"],
        task_prompt=task_prompt,
        work_dir=work_dir,
        model_name=model_name,
        eval_timeout=eval_timeout,
        model_map=MODEL_MAP,
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
