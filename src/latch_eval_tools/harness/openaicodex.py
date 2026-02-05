import os
from pathlib import Path
from latch_eval_tools.harness._cli_runner import _run_cli_agent, EVAL_TIMEOUT

MODEL_MAP = {
    "openai/gpt-5.3-codex": "gpt-5.3-codex",
    "openai/gpt-5.2-codex": "gpt-5.2-codex",
    "openai/gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
    "openai/gpt-5.1-codex-max": "gpt-5.1-codex-max",
    "openai/gpt-5.2": "gpt-5.2",
    "openai/gpt-5.1": "gpt-5.1",
    "openai/gpt-5.1-codex": "gpt-5.1-codex",
    "openai/gpt-5-codex": "gpt-5-codex",
    "openai/gpt-5-codex-mini": "gpt-5-codex-mini",
    "openai/gpt-5": "gpt-5",
    "openai/gpt-4o": "gpt-4o",
    "openai/o1": "o1",
    "openai/o1-mini": "o1-mini",
}


def run_openaicodex_task(
    task_prompt: str,
    work_dir: Path,
    model_name: str | None = None,
    eval_timeout: int = EVAL_TIMEOUT,
) -> dict:
    openai_key = os.environ.get("OPENAI_API_KEY")
    codex_key = os.environ.get("CODEX_API_KEY")

    if openai_key is None and codex_key is None:
        raise ValueError("OPENAI_API_KEY or CODEX_API_KEY environment variable is required for OpenAI Codex")

    if openai_key is not None and codex_key is None:
        os.environ["CODEX_API_KEY"] = openai_key

    return _run_cli_agent(
        agent_type="openaicodex",
        cli_command=["codex", "exec"],
        task_prompt=task_prompt,
        work_dir=work_dir,
        model_name=model_name,
        eval_timeout=eval_timeout,
        model_map=MODEL_MAP,
    )
