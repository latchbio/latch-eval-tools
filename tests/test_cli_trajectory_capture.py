import io
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from latch_eval_tools.harness import _cli_runner


def _run_claude(
    monkeypatch: pytest.MonkeyPatch,
    work_dir: Path,
    stdout: Iterable[str],
    stderr: str = "",
) -> dict[str, Any]:
    monkeypatch.delenv(_cli_runner.FALLBACK_API_KEYS_ENV, raising=False)
    monkeypatch.setattr(_cli_runner, "ensure_docker_image", lambda _: None)
    monkeypatch.setattr(_cli_runner, "get_agent_workspace_dir", lambda _: work_dir)
    monkeypatch.setattr(
        _cli_runner, "_create_cli_container", lambda **_: "/root/.claude"
    )
    monkeypatch.setattr(_cli_runner, "_start_cli_container", lambda _: None)
    monkeypatch.setattr(_cli_runner, "teardown_container", lambda _: None)
    process = SimpleNamespace(
        stdin=io.StringIO(),
        stdout=stdout,
        stderr=io.StringIO(stderr),
        returncode=0,
        poll=lambda: 0,
    )
    monkeypatch.setattr(_cli_runner.subprocess, "Popen", lambda *_, **__: process)
    (work_dir / "finished.txt").write_text("done")
    return _cli_runner._run_cli_agent(
        agent_type="claudecode",
        cli_command=["claude"],
        task_prompt="Complete the task.",
        work_dir=work_dir,
        memory_limit_bytes=1024,
        completion=True,
    )


def test_live_snapshots_are_throttled_and_final_snapshot_is_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = [
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": f"Message {index}"},
        }
        for index in range(6)
    ]
    events.append({"type": "result", "subtype": "success", "num_turns": 6})
    event_times = [0.1, 0.2, 0.3, 1.1, 1.2, 2.2, 2.3]
    now = 0.0
    monkeypatch.setattr(_cli_runner.time, "monotonic", lambda: now)

    snapshots = []
    write_text = Path.write_text

    def record_snapshot(path: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if path == tmp_path / "trajectory.json":
            snapshots.append((now, json.loads(data)))
        return write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", record_snapshot)

    def stdout() -> Iterable[str]:
        nonlocal now
        for event_time, event in zip(event_times, events):
            now = event_time
            yield json.dumps(event) + "\n"

    result = _run_claude(monkeypatch, tmp_path, stdout())

    assert snapshots[0] == (0.0, [])
    assert snapshots[1:-1] == [(1.1, events[:4]), (2.2, events[:6])]
    assert snapshots[-1] == (2.3, events)
    assert json.loads((tmp_path / "trajectory.json").read_text()) == events
    assert result["answer"]["last_message"] == "Message 5"


def test_finalization_waits_for_stdout_and_stderr_to_drain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class DelayedReader:
        """Simulate a reader requiring six seconds, without a wall-clock sleep."""

        def __init__(self, target: Callable[[], None], daemon: bool) -> None:
            self.target = target

        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            if timeout is None or timeout >= 6:
                self.target()

    monkeypatch.setattr(_cli_runner.threading, "Thread", DelayedReader)
    events = [
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": "Final answer"},
        },
        {"type": "result", "subtype": "success", "num_turns": 1},
    ]
    stdout = "".join(json.dumps(event) + "\n" for event in events)

    result = _run_claude(
        monkeypatch, tmp_path, io.StringIO(stdout), stderr="Final diagnostic\n"
    )

    assert json.loads((tmp_path / "trajectory.json").read_text()) == events
    assert result["answer"]["last_message"] == "Final answer"
    assert (tmp_path / "agent_output.log").read_text() == (
        stdout + "\n\nSTDERR:\nFinal diagnostic\n"
    )
