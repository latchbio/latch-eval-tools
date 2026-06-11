"""The ``latch-eval run`` subcommand: run one eval JSON and report the verdict."""

import json
import sys
from pathlib import Path

from latch_eval_tools.cli.harnesses import build_agent_function
from latch_eval_tools.cli.preflight import preflight


def grader_result_to_dict(grader_result):
    if grader_result is None:
        return None
    return {
        "passed": grader_result.passed,
        "score": getattr(grader_result, "score", None),
        "reasoning": grader_result.reasoning,
        "metrics": grader_result.metrics,
        "field_scores": getattr(grader_result, "field_scores", {}),
    }


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

    if not args.no_preflight:
        # When --data is supplied it overrides the eval's data_node, so no Latch
        # download (and no token) is needed.
        preflight(args.harness, args.docker_image, needs_latch_token=not args.data)

    from latch_eval_tools import EvalRunner

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

    agent_function = build_agent_function(
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
            "grader_result": grader_result_to_dict(result.get("grader_result")),
            "metadata": metadata,
        }
        out_path = Path(args.json_out).expanduser()
        out_path.write_text(json.dumps(out, indent=2, default=str))
        print(f"  json:    {out_path}")

    # Exit non-zero on FAIL/NO-GRADE so callers (and the run-eval agent) can branch.
    return 0 if passed else 1
