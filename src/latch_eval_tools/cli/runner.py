"""The ``latch-eval run`` subcommand: run one eval JSON and report the verdict."""

import json
import sys
import shutil
import os
from pathlib import Path

from latch_eval_tools.harness.runner import EvalRunner
from latch_eval_tools.harness import (
    run_claudecode_task,
    run_minisweagent_task,
    run_openaicodex_task,
    run_pi_task,
)

# List of supported harnesses
HARNESSES = ("claudecode", "minisweagent", "openaicodex", "pi")

# Env vars the harnesses need. Used only for warning displays.
HARNESS_ENV_HINTS = {
    "claudecode": ["ANTHROPIC_API_KEY"],
    "minisweagent": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"],
    "openaicodex": ["OPENAI_API_KEY", "CODEX_API_KEY"],
    "pi": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "FIREWORKS_API_KEY"],
}


def _grader_result_to_dict(grader_result):
    if grader_result is None:
        return None
    return {
        "passed": grader_result.passed,
        "score": getattr(grader_result, "score", None),
        "reasoning": grader_result.reasoning,
        "metrics": grader_result.metrics,
        "field_scores": getattr(grader_result, "field_scores", {}),
    }


def _build_agent_function(harness, model, eval_timeout, docker_image):
    """Return an ``agent_function(task_prompt, work_dir)`` for the chosen harness."""
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


def run_command(args):
    eval_path = Path(args.eval).expanduser().resolve()
    if not eval_path.exists():
        raise SystemExit(f"Eval file not found: {eval_path}")

    # EvalRunner -> Eval(...) only consumes a single `grader`. Warn loudly if the
    # JSON uses a multi-grader `graders` list, which this path cannot grade.
    try:
        raw = json.loads(eval_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Eval file is not valid JSON: {exc}")
    if isinstance(raw, dict) and raw.get("graders") and not raw.get("grader"):
        print(
            "[warn] This eval uses a `graders` list. The local runner only grades "
            "a single `grader`; the agent will run but no grade will be produced. "
            "Use the platform judge for multi-grader evals.",
            file=sys.stderr,
        )
    
    # Run some non-fatal preflight checks
    if not args.no_preflight:
        warnings = []
        if shutil.which("docker") is None:
            warnings.append(
                "Docker not found on PATH. All harnesses run the agent inside a "
                "Docker container; the run will fail without it."
            )
        hints = HARNESS_ENV_HINTS.get(args.harness, [])
        if hints and not any(os.environ.get(k) for k in hints):
            warnings.append(
                f"None of {hints} is set; the {args.harness} harness needs one to reach "
                "its model provider."
            )
        if not args.data and not(Path.home() / ".latch" / "token").exists():
            warnings.append(
                "No Latch token at ~/.latch/token; data_node downloads will fail. "
                "Run `latch login` (or pass --data to use local files)."
            )
        for w in warnings:
            print(f"[preflight] WARNING: {w}", file=sys.stderr)

    runner = EvalRunner(
        eval_path,
        keep_workspace=args.keep_workspace,
        run_id=args.run_id,
        benchmark_name=args.benchmark_name,
        data_node_override=args.data,
        # Anchor workspaces to the invocation directory (--output-dir is the runs
        # dir itself, so workspace_name="" places eval dirs directly inside it),
        # and the dataset cache to a shared user-level location.
        work_root=Path(args.output_dir).expanduser().resolve(),
        workspace_name="",
        cache_dir=Path(args.cache_dir).expanduser(),
    )

    agent_function = _build_agent_function(
        args.harness, args.model, args.eval_timeout, args.docker_image
    )

    result = runner.run(agent_function=agent_function)

    print("\n" + "=" * 80)
    print("RUN SUMMARY")
    print("=" * 80)
    passed = result.get("passed")
    verdict = "PASS" if passed else ("FAIL" if passed is False else "NO GRADE")
    print(f"  eval:    {result.get('test_id')}")
    print(f"  harness: {args.harness}" + (f" ({args.model})" if args.model else ""))
    print(f"  verdict: {verdict}")
    metadata = result.get("metadata") or {}
    if metadata.get("error_details"):
        print(f"  error:   {metadata['error_details'].get('error')}")

    if args.json_out:
        out = {
            "test_id": result.get("test_id"),
            "harness": args.harness,
            "model": args.model,
            "passed": passed,
            "agent_answer": result.get("agent_answer"),
            "grader_result": _grader_result_to_dict(result.get("grader_result")),
            "metadata": metadata,
        }
        out_path = Path(args.json_out).expanduser()
        out_path.write_text(json.dumps(out, indent=2, default=str))
        print(f"  json:    {out_path}")

    # Exit non-zero on FAIL/NO-GRADE so callers (and the run-eval agent) can branch.
    return 0 if passed else 1
