#!/usr/bin/env python3
"""Full-harness smoke run for the grok-build harness.

Runs `run_grokbuild_task` (inside the agent_env Docker image) against the tiny
self-contained eval in examples/grok_smoke_eval.json and prints the answer,
run-summary metadata, and a tail of the parsed trajectory. This exercises the
whole path: docker exec -> grok headless -> streaming-json parse -> answer file.

Prereqs (this environment currently has NEITHER):
  - Docker daemon running
  - export XAI_API_KEY=xai-...
  - agent_env image built and tagged (see DEFAULT_DOCKER_IMAGE), e.g.:
      docker build -t <default-image-tag> agent_env/
    Run examples/grok_smoke_capture.sh FIRST to verify grok's event schema,
    then reconcile _GROK_USAGE_ALIASES / AGENT_IDENTIFIER_KEYS if needed.

Usage:
  uv run python examples/run_grokbuild_eval.py [path/to/eval.json] [model]
"""
import json
import os
import sys
from pathlib import Path

from latch_eval_tools import EvalRunner, run_grokbuild_task

EVAL_PATH = sys.argv[1] if len(sys.argv) > 1 else "examples/grok_smoke_eval.json"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "xai/grok-4.6"
# The published DEFAULT_DOCKER_IMAGE doesn't have grok yet (and the full
# agent_env build currently fails at `uv sync`). For a harness smoke test that
# doesn't need the bio stack, point at the lightweight grok-only image:
#   docker build -f examples/Dockerfile.grok-smoke -t grok-smoke .
#   GROK_IMAGE=grok-smoke uv run python examples/run_grokbuild_eval.py
GROK_IMAGE = os.environ.get("GROK_IMAGE")  # None -> harness default image


def main() -> int:
    if not os.environ.get("XAI_API_KEY"):
        print("ERROR: XAI_API_KEY is not set. export XAI_API_KEY=xai-... first.")
        return 1

    runner = EvalRunner(
        EVAL_PATH,
        keep_workspace=True,  # keep so we can inspect trajectory.json / logs
        benchmark_name="grok-build smoke",
    )
    def agent(task, work_dir):
        kwargs = dict(
            task_prompt=task,
            work_dir=work_dir,
            model_name=MODEL,
            eval_timeout=300,
            benchmark=True,
        )
        if GROK_IMAGE:
            kwargs["docker_image"] = GROK_IMAGE
        return run_grokbuild_task(**kwargs)

    result = runner.run(agent_function=agent)

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print("answer:", json.dumps(result.get("agent_answer"), indent=2))
    meta = result.get("metadata", {}) or {}
    run_summary = meta.get("run_summary")
    print("\nmodel:", meta.get("model"))
    print("duration_s:", meta.get("duration_s"))
    print("usage:", meta.get("usage"))
    print("n_steps:", meta.get("n_steps"))
    print("timed_out:", meta.get("timed_out"))
    if meta.get("error_details"):
        print("error_details:", json.dumps(meta["error_details"], indent=2)[:1500])
    if run_summary:
        print("run_summary.metrics:", json.dumps(run_summary.get("metrics"), indent=2))
        print("run_summary.refusal.status:",
              (run_summary.get("refusal") or {}).get("status"))

    # Was the harness able to parse the streamed events at all?
    ok = (
        result.get("agent_answer") is not None
        and not meta.get("timed_out")
        and not meta.get("error_details")
    )
    print("\nSMOKE:", "PASS" if ok else "FAIL (inspect trajectory.json + agent_output.log)")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
