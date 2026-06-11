"""Harness selection and agent-function construction for the ``latch-eval`` CLI."""

HARNESSES = ("claudecode", "minisweagent", "openaicodex", "pi")

# Env var each harness needs in order to talk to its model provider. Used only
# for a friendly preflight warning; the harness itself is the source of truth.
HARNESS_ENV_HINTS = {
    "claudecode": ["ANTHROPIC_API_KEY"],
    "minisweagent": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"],
    "openaicodex": ["OPENAI_API_KEY", "CODEX_API_KEY"],
    "pi": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "FIREWORKS_API_KEY"],
}


def build_agent_function(harness, model, eval_timeout, docker_image):
    """Return an ``agent_function(task_prompt, work_dir)`` for the chosen harness."""
    from latch_eval_tools import (
        run_claudecode_task,
        run_minisweagent_task,
        run_openaicodex_task,
        run_pi_task,
    )

    common = {}
    if eval_timeout is not None:
        common["eval_timeout"] = eval_timeout
    if docker_image:
        common["docker_image"] = docker_image

    if harness == "claudecode":
        return lambda task, wd: run_claudecode_task(task, wd, model_name=model, **common)
    if harness == "minisweagent":
        if not model:
            raise SystemExit(
                "minisweagent requires --model "
                "(e.g. --model anthropic/claude-sonnet-4-6)"
            )
        return lambda task, wd: run_minisweagent_task(task, wd, model_name=model, **common)
    if harness == "openaicodex":
        return lambda task, wd: run_openaicodex_task(task, wd, model_name=model, **common)
    if harness == "pi":
        return lambda task, wd: run_pi_task(task, wd, model_name=model, **common)
    raise SystemExit(f"Unknown harness: {harness}")
