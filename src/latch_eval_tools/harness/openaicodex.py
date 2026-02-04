from pathlib import Path
from latch_eval_tools.harness._cli_runner import _run_cli_agent, EVAL_TIMEOUT

MODEL_MAP = {
    "openai/gpt-5-codex": "gpt-5-codex",
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
    return _run_cli_agent(
        agent_type="openaicodex",
        cli_command=["codex", "exec"],
        task_prompt=task_prompt,
        work_dir=work_dir,
        model_name=model_name,
        eval_timeout=eval_timeout,
        model_map=MODEL_MAP,
    )
