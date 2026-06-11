"""Argument parsing and entry point for the ``latch-eval`` CLI.

``latch-eval run`` wraps :class:`~latch_eval_tools.EvalRunner` so a single drafted
eval JSON can be run end-to-end (download data -> run an agent in a Docker sandbox
-> grade the answer) from the shell:

    latch-eval run --eval evals/QC01.json --harness claudecode
    latch-eval run -e evals/DE03.json --harness minisweagent --model anthropic/claude-sonnet-4-6

This is the fast local feedback loop for eval authoring: draft a JSON, run it
against a real agent, read the grader's verdict and the agent trajectory.
"""

import argparse

from latch_eval_tools.cli.harnesses import HARNESSES
from latch_eval_tools.cli.run import run_command


def build_parser():
    parser = argparse.ArgumentParser(
        prog="latch-eval",
        description="Run and grade Latch eval JSON files locally.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a single eval JSON against an agent and grade it.")
    run_p.add_argument("-e", "--eval", required=True, help="Path to the eval JSON file.")
    run_p.add_argument(
        "--harness",
        required=True,
        choices=HARNESSES,
        help="Agent harness to run the task with.",
    )
    run_p.add_argument(
        "--model",
        default=None,
        help="Model name for the harness (required for minisweagent, "
        "e.g. anthropic/claude-sonnet-4-6).",
    )
    run_p.add_argument(
        "--data",
        action="append",
        default=None,
        metavar="PATH",
        help="Local file or directory to stage as the agent's /workspace/data, "
        "bypassing the eval's data_node download entirely. Repeatable. Use when "
        "you already have the data on disk.",
    )
    run_p.add_argument("--run-id", default=None, help="Optional run ID to namespace the workspace.")
    run_p.add_argument(
        "--eval-timeout",
        type=int,
        default=None,
        help="Override the agent eval timeout in seconds.",
    )
    run_p.add_argument(
        "--docker-image",
        default=None,
        help="Override the Docker image used to sandbox the agent.",
    )
    run_p.add_argument(
        "--keep-workspace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the workspace (trajectory.json, agent_output.log, eval_answer.json) "
        "after the run. On by default for the authoring loop.",
    )
    run_p.add_argument("--cache-name", default=".eval_cache", help="Dataset cache directory name.")
    run_p.add_argument("--workspace-name", default=".eval_workspace", help="Workspace directory name.")
    run_p.add_argument("--benchmark-name", default="Eval", help="Display name for the benchmark.")
    run_p.add_argument("--json-out", default=None, help="Write a structured result JSON to this path.")
    run_p.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip the Docker / API key / Latch token preflight checks.",
    )
    run_p.set_defaults(func=run_command)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
