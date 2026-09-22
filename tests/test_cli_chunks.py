import json
import os
from pathlib import Path
from typing import Any

import pytest

from latch_eval_tools.harness import _cli_runner, run_claudecode_chunk, run_pi_chunk
from latch_eval_tools.harness.run_summary import build_cli_run_summary

PI_ABORTED_TURN_END = {
    "type": "turn_end",
    "message": {"role": "assistant", "stopReason": "aborted", "content": []},
    "toolResults": [],
}


def _fake_docker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    events: list[dict[str, Any]],
    returncode: int,
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_file = tmp_path / "argv.json"
    stdout = "".join(f"{json.dumps(event)}\n" for event in events)
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(argv_file)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
        "sys.stdin.read()\n"
        f"sys.stdout.write({stdout!r})\n"
        f"sys.exit({returncode})\n"
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return argv_file


def test_default_commands_have_no_chunk_flags() -> None:
    claude = _cli_runner._build_agent_command(
        "claudecode", ["claude"], None, None, None, resume_identifier="session-id"
    )
    pi = _cli_runner._build_agent_command(
        "pi", ["pi"], None, None, None, resume_identifier="session-id"
    )

    assert "--fork-session" not in claude
    assert "--max-turns" not in claude
    assert pi[pi.index("--session") + 1] == "session-id"
    assert "--fork" not in pi
    assert "--max-turns" not in pi
    assert _cli_runner.PI_MAX_TURNS_EXTENSION_CONTAINER_PATH not in pi


def test_claude_fork_command_resumes_into_a_new_session() -> None:
    command = _cli_runner._build_agent_command(
        "claudecode",
        ["claude"],
        None,
        None,
        None,
        resume_identifier="session-id",
        max_turns=3,
        fork=True,
    )

    assert command[command.index("--resume") + 1] == "session-id"
    assert command[command.index("--resume") + 2] == "--fork-session"
    assert command[command.index("--max-turns") + 1] == "3"


def test_pi_fork_command_loads_the_max_turns_extension() -> None:
    command = _cli_runner._build_agent_command(
        "pi",
        ["pi"],
        None,
        None,
        None,
        resume_identifier="session-id",
        max_turns=3,
        fork=True,
    )

    assert command[command.index("--fork") + 1] == "session-id"
    assert "--session" not in command
    extension_index = command.index(_cli_runner.PI_MAX_TURNS_EXTENSION_CONTAINER_PATH)
    assert command[extension_index - 1] == "--extension"
    assert command[command.index("--max-turns") + 1] == "3"


def test_pi_turn_count_ignores_the_aborted_turn() -> None:
    summary = build_cli_run_summary(
        agent_type="pi",
        trajectory=[{"type": "turn_end"}, {"type": "turn_end"}, PI_ABORTED_TURN_END],
        duration_seconds=1.0,
        model_name=None,
    )

    assert summary.metrics.turn_count == 2


def test_claude_chunk_treats_max_turns_as_a_normal_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session_file = tmp_path / ".claude" / "projects" / "-workspace" / "session-1.jsonl"
    session_file.parent.mkdir(parents=True)
    session_file.touch()
    argv_file = _fake_docker(
        monkeypatch,
        tmp_path,
        [
            {"type": "system", "subtype": "init", "session_id": "session-1"},
            {
                "type": "assistant",
                "message": {"id": "msg-1"},
                "parent_tool_use_id": None,
            },
            {
                "type": "assistant",
                "message": {"id": "msg-1"},
                "parent_tool_use_id": None,
            },
            {
                "type": "assistant",
                "message": {"id": "sub-1"},
                "parent_tool_use_id": "tool-1",
            },
            {
                "type": "assistant",
                "message": {"id": "msg-2"},
                "parent_tool_use_id": None,
            },
            {
                "type": "result",
                "subtype": "error_max_turns",
                "num_turns": 3,
                "session_id": "session-1",
            },
        ],
        returncode=1,
    )

    chunk = run_claudecode_chunk(
        "container-a",
        "task",
        tmp_path,
        2,
        model_name="anthropic/claude-fable-5",
    )

    assert chunk.session_id == "session-1"
    assert chunk.session_path == "/root/.claude/projects/-workspace/session-1.jsonl"
    assert chunk.turns == 2
    assert chunk.hit_turn_limit
    assert len(chunk.events) == 6
    assert json.loads((tmp_path / "trajectory.json").read_text()) == chunk.events
    argv = json.loads(argv_file.read_text())
    assert argv[:3] == ["exec", "-i", "container-a"]
    assert argv[argv.index("--model") + 1] == "claude-fable-5"
    assert argv[argv.index("--max-turns") + 1] == "2"


def test_claude_chunk_raises_on_a_failed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_docker(
        monkeypatch,
        tmp_path,
        [
            {
                "type": "result",
                "subtype": "success",
                "is_error": True,
                "session_id": "session-1",
            }
        ],
        returncode=1,
    )

    with pytest.raises(RuntimeError, match="exited with code 1"):
        run_claudecode_chunk("container-a", "task", tmp_path, 2)


def test_pi_chunk_forks_and_ignores_the_aborted_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sessions_dir = tmp_path / ".pi" / "agent" / "sessions" / "--workspace--"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "2026-09-22T00-00-00-000Z_session-2.jsonl").touch()
    argv_file = _fake_docker(
        monkeypatch,
        tmp_path,
        [
            {"type": "session", "id": "session-2"},
            {"type": "message_update"},
            {"type": "turn_end", "message": {"role": "assistant", "content": []}},
            PI_ABORTED_TURN_END,
        ],
        returncode=0,
    )

    chunk = run_pi_chunk(
        "container-b",
        "Continue.",
        tmp_path,
        1,
        resume_identifier="session-1",
        fork=True,
    )

    assert chunk.session_id == "session-2"
    assert chunk.session_path == (
        "/root/.pi/agent/sessions/--workspace--/2026-09-22T00-00-00-000Z_session-2.jsonl"
    )
    assert chunk.turns == 1
    assert chunk.hit_turn_limit
    assert [event["type"] for event in chunk.events] == [
        "session",
        "turn_end",
        "turn_end",
    ]
    assert (tmp_path / ".pi" / "max_turns.js").exists()
    assert (tmp_path / ".pi" / "tool_timeout.js").exists()
    argv = json.loads(argv_file.read_text())
    assert argv[argv.index("--fork") + 1] == "session-1"
    assert argv[argv.index("--max-turns") + 1] == "1"
