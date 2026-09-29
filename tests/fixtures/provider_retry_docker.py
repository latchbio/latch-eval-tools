"""Subprocess stand-in for Docker, used only by test_cli_provider_recovery.py."""

import json
import os
import sys
from pathlib import Path


def main() -> int:
    root = Path(os.environ["TEST_PROVIDER_RECOVERY_ROOT"])
    args = sys.argv[1:]
    with (root / "docker_calls.jsonl").open("a") as stream:
        stream.write(json.dumps(args) + "\n")
    if args[0] == "inspect":
        print("true" if args[2] == "{{.State.Running}}" else "false")
        return 0
    if args[0] != "exec":
        return 0

    attempts = json.loads((root / "attempts.json").read_text())
    counter = root / "attempt_count"
    index = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(index + 1))
    attempt = attempts[index]
    workspace = Path.cwd()
    session_file = workspace.parent / ".pi/agent/sessions/--workspace--/session-1.jsonl"
    with (root / "prompts.jsonl").open("a") as stream:
        stream.write(json.dumps(sys.stdin.read()) + "\n")
    # Work completed before the first failed API call must survive every resume.
    if index == 0:
        (workspace / "intermediate.bin").write_bytes(b"completed analysis")
        (root / "tool_executions").write_text("1")
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.write_text(
            '{"type": "toolResult", "toolCallId": "completed-tool"}\n'
        )
    else:
        assert (workspace / "intermediate.bin").read_bytes() == b"completed analysis"
        assert json.loads(session_file.read_text())["toolCallId"] == "completed-tool"
        resume_flag = "--session" if "pi" in args else "--resume"
        assert args[args.index(resume_flag) + 1] == "session-1"
    if attempt.get("finish"):
        (workspace / "eval_answer.json").write_text('{"answer": 42}')
    for event in attempt["events"]:
        print(json.dumps(event), flush=True)
    return attempt.get("returncode", 0)


if __name__ == "__main__":
    sys.exit(main())
