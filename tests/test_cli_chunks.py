import json
import os
from pathlib import Path
from typing import Any

import pytest

from latch_eval_tools.harness import (
    _cli_runner,
    claudecode,
    run_claudecode_chunk,
    run_pi_chunk,
)

PI_ABORTED_TURN_END = {
    "type": "turn_end",
    "message": {"role": "assistant", "stopReason": "aborted", "content": []},
    "toolResults": [],
}
PI_ERROR_TURN_END = {
    "type": "turn_end",
    "message": {
        "role": "assistant",
        "stopReason": "error",
        "content": [],
        "errorMessage": "529 overloaded",
    },
    "toolResults": [],
}
PI_TOOL_TURN_END = {
    "type": "turn_end",
    "message": {"role": "assistant", "stopReason": "toolUse", "content": []},
}


def _fake_docker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    events: list[dict[str, Any]],
    returncode: int,
    sleep_seconds: int = 0,
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_file = tmp_path / "docker_calls.jsonl"
    stdout = "".join(f"{json.dumps(event)}\n" for event in events)
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, time\n"
        f"open({str(calls_file)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[-1] == 'kill -9 -1':\n"
        "    sys.exit(0)\n"
        "sys.stdin.read()\n"
        f"sys.stdout.write({stdout!r})\n"
        "sys.stdout.flush()\n"
        f"time.sleep({sleep_seconds})\n"
        f"sys.exit({returncode})\n"
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return calls_file


def _docker_calls(calls_file: Path) -> list[list[str]]:
    return [json.loads(line) for line in calls_file.read_text().splitlines()]


def test_pi_resume_command_keeps_the_session_flag() -> None:
    command = _cli_runner._build_agent_command(
        "pi", ["pi"], None, None, None, resume_identifier="session-id"
    )

    assert command[command.index("--session") + 1] == "session-id"


def test_claude_fork_command_resumes_into_a_new_session() -> None:
    command = _cli_runner._build_agent_command(
        "claudecode",
        ["claude"],
        None,
        None,
        None,
        resume_identifier="session-id",
        fork=True,
    )

    assert command[command.index("--resume") + 1] == "session-id"
    assert command[command.index("--resume") + 2] == "--fork-session"


@pytest.mark.parametrize(
    ("resume_identifier", "expected_prompt"),
    [(None, f"task\n{claudecode.BACKGROUND_PROCESS_NOTE}"), ("session-1", "task")],
)
def test_claude_chunk_adds_the_background_note_only_to_a_new_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resume_identifier: str | None,
    expected_prompt: str,
) -> None:
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        claudecode, "_run_cli_chunk", lambda **kwargs: observed.update(kwargs)
    )

    run_claudecode_chunk(
        "container-a", "task", tmp_path, 2, resume_identifier=resume_identifier
    )

    assert observed["prompt"] == expected_prompt


def test_claude_chunk_treats_max_turns_as_a_normal_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session_file = tmp_path / ".claude" / "projects" / "-workspace" / "session-1.jsonl"
    session_file.parent.mkdir(parents=True)
    session_file.touch()
    calls_file = _fake_docker(
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
        switch_models_on_flag=True,
    )

    assert chunk.session_id == "session-1"
    assert chunk.session_file == session_file
    assert chunk.turns == 2
    assert chunk.hit_turn_limit
    assert len(json.loads((tmp_path / "trajectory.json").read_text())) == 6
    [argv] = _docker_calls(calls_file)
    assert argv[:6] == [
        "exec",
        "-i",
        "container-a",
        "env",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1",
        "claude",
    ]
    assert argv[argv.index("--model") + 1] == "claude-fable-5"
    assert argv[argv.index("--max-turns") + 1] == "2"
    assert json.loads(argv[argv.index("--settings") + 1]) == {
        "switchModelsOnFlag": True
    }


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
                "result": "API Error: 401",
                "session_id": "session-1",
            }
        ],
        returncode=1,
    )

    with pytest.raises(RuntimeError, match="exit code 1: API Error: 401"):
        run_claudecode_chunk("container-a", "task", tmp_path, 2)


def test_pi_chunk_raises_on_an_error_turn_despite_exit_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_docker(
        monkeypatch,
        tmp_path,
        [{"type": "session", "id": "session-1"}, PI_ERROR_TURN_END],
        returncode=0,
    )

    with pytest.raises(RuntimeError, match="529 overloaded"):
        run_pi_chunk("container-b", "task", tmp_path, 2)


def test_pi_chunk_cut_by_output_length_is_not_a_natural_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sessions_dir = tmp_path / ".pi" / "agent" / "sessions" / "--workspace--"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "2026-09-22T00-00-00-000Z_session-1.jsonl").touch()
    _fake_docker(
        monkeypatch,
        tmp_path,
        [
            {"type": "session", "id": "session-1"},
            PI_TOOL_TURN_END,
            {
                "type": "turn_end",
                "message": {"role": "assistant", "stopReason": "length", "content": []},
            },
        ],
        returncode=0,
    )

    chunk = run_pi_chunk("container-b", "task", tmp_path, 5)

    assert chunk.turns == 2
    assert chunk.hit_turn_limit


def test_chunk_timeout_kills_the_agent_inside_the_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls_file = _fake_docker(monkeypatch, tmp_path, [], returncode=0, sleep_seconds=30)

    with pytest.raises(RuntimeError, match="^pi chunk timed out after 1s$"):
        run_pi_chunk("container-b", "task", tmp_path, 2, timeout=1)

    assert _docker_calls(calls_file)[-1] == [
        "exec",
        "container-b",
        "sh",
        "-c",
        "kill -9 -1",
    ]


def test_pi_chunk_forks_and_ignores_the_aborted_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sessions_dir = tmp_path / ".pi" / "agent" / "sessions" / "--workspace--"
    sessions_dir.mkdir(parents=True)
    session_file = sessions_dir / "2026-09-22T00-00-00-000Z_session-2.jsonl"
    session_file.touch()
    calls_file = _fake_docker(
        monkeypatch,
        tmp_path,
        [
            {"type": "session", "id": "session-2"},
            {"type": "message_update"},
            PI_TOOL_TURN_END,
            PI_ERROR_TURN_END,
            PI_TOOL_TURN_END,
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
    assert chunk.session_file == session_file
    assert chunk.turns == 2
    assert chunk.hit_turn_limit
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert [event["type"] for event in trajectory] == ["session"] + ["turn_end"] * 4
    assert (tmp_path / ".pi" / "max_turns.js").exists()
    assert (tmp_path / ".pi" / "tool_timeout.js").exists()
    [argv] = _docker_calls(calls_file)
    assert argv[argv.index("--fork") + 1] == "session-1"
    assert "--session" not in argv
    extension_index = argv.index(_cli_runner.PI_MAX_TURNS_EXTENSION_CONTAINER_PATH)
    assert argv[extension_index - 1] == "--extension"
    assert argv[argv.index("--max-turns") + 1] == "1"
