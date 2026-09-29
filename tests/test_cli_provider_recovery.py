import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from latch_eval_tools.harness import _cli_runner, run_claudecode_task, run_pi_task


def pi_message(stop_reason: str, *, error: str = "") -> dict[str, Any]:
    return {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "stopReason": stop_reason,
            "errorMessage": error,
            "content": [{"type": "text", "text": "response"}],
            "usage": {"input": 10, "output": 2, "cost": {"total": 0.01}},
        },
    }


def failure(*, status: int = 429, progress: bool = False) -> dict[str, Any]:
    return {
        "events": [
            {"type": "session", "id": "session-1"},
            *([pi_message("toolUse")] if progress else []),
            pi_message("error", error=f"{status}: provider failed"),
        ]
    }


SUCCESS = {"finish": True, "events": [pi_message("stop")]}


@dataclass
class RecoveryRun:
    root: Path
    elapsed: float = 0.0
    waits: list[float] = field(default_factory=list)
    extra_wait: float = 0.0
    real_sleep: Callable[[float], None] = time.sleep

    def sleep(self, seconds: float) -> None:
        if seconds <= 1:
            # Let the real child process drain its pipes. Polling time does not
            # affect the deterministic provider cooldown clock in these tests.
            self.real_sleep(0.001)
        else:
            self.waits.append(seconds)
            self.elapsed += seconds + self.extra_wait

    def monotonic(self) -> float:
        return self.elapsed

    def run(
        self,
        attempts: list[dict[str, Any]],
        *,
        wait_budget: float = 1800,
        timeout: int = 6000,
        agent: str = "pi",
    ) -> dict[str, Any]:
        (self.root / "attempts.json").write_text(json.dumps(attempts))
        work_dir = self.root / "work"
        work_dir.mkdir()
        (work_dir / "agent_workspace").mkdir()
        run = run_pi_task if agent == "pi" else run_claudecode_task
        return run(
            "Original task",
            work_dir,
            docker_image="test-image",
            memory_limit_bytes=1024,
            prompt_suffix=None,
            completion_file_path="eval_answer.json",
            provider_retry_wait_seconds=wait_budget,
            eval_timeout=timeout,
        )

    def calls(self, operation: str) -> list[list[str]]:
        return [
            args
            for line in (self.root / "docker_calls.jsonl").read_text().splitlines()
            if (args := json.loads(line))[0] == operation
        ]

    def assert_completed(self, result: dict[str, Any]) -> None:
        # With an explicit completion path, V2 reads/validates the file itself.
        assert "error_details" not in result["metadata"]
        output = self.root / "work/agent_workspace/eval_answer.json"
        assert json.loads(output.read_text()) == {"answer": 42}


@pytest.fixture
def recovery_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> RecoveryRun:
    fixture = Path(__file__).parent / "fixtures" / "provider_retry_docker.py"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(f"#!{sys.executable}\n" + fixture.read_text())
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("TEST_PROVIDER_RECOVERY_ROOT", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.delenv(_cli_runner.FALLBACK_API_KEYS_ENV, raising=False)
    monkeypatch.setattr(_cli_runner, "ensure_docker_image", lambda _image: None)
    monkeypatch.setattr(_cli_runner.random, "uniform", lambda _start, _end: 0.0)
    run = RecoveryRun(tmp_path)
    monkeypatch.setattr(_cli_runner.time, "sleep", run.sleep)
    monkeypatch.setattr(_cli_runner.time, "monotonic", run.monotonic)
    return run


def test_more_than_five_provider_failures_resume_in_place(
    recovery_run: RecoveryRun,
) -> None:
    result = recovery_run.run([failure(progress=True)] * 7 + [SUCCESS])
    recovery_run.assert_completed(result)
    assert result["metadata"]["provider_retry_count"] == 7
    assert result["metadata"]["provider_retry_wait_seconds"] == 420
    assert recovery_run.waits == [60] * 7
    assert len(recovery_run.calls("create")) == 1
    assert len(recovery_run.calls("start")) == 1
    assert len(recovery_run.calls("rm")) == 1
    execs = recovery_run.calls("exec")
    assert len(execs) == 8
    assert len({args[2] for args in execs}) == 1
    assert "--session" not in execs[0]
    assert all(args[args.index("--session") + 1] == "session-1" for args in execs[1:])
    prompts = [
        json.loads(line)
        for line in (recovery_run.root / "prompts.jsonl").read_text().splitlines()
    ]
    assert prompts == ["Original task"] + ["Continue."] * 7
    assert (recovery_run.root / "tool_executions").read_text() == "1"
    trajectory = json.loads((recovery_run.root / "work/trajectory.json").read_text())
    assert len([event for event in trajectory if event["type"] == "message_end"]) == 15
    assert result["metadata"]["run_summary"]["metrics"]["usage"]["input_tokens"] == 150


def test_backoff_resets_after_progress_without_resetting_budget(
    recovery_run: RecoveryRun,
) -> None:
    result = recovery_run.run(
        [failure(), failure(), failure(progress=True), failure()], wait_budget=250
    )
    assert recovery_run.waits == [60, 120, 60]
    assert len(recovery_run.calls("exec")) == 4
    details = result["metadata"]["error_details"]
    assert details["api_error_status"] == 429
    assert details["provider_retry_stop_reason"] == "wait_budget_exhausted"
    assert details["provider_retry_wait_seconds"] == 240
    assert details["provider_retry_count"] == 3
    assert not details["timed_out"]


def test_no_progress_exhausts_wait_budget(recovery_run: RecoveryRun) -> None:
    result = recovery_run.run([failure()] * 10)
    assert recovery_run.waits == [60, 120, 240, 300, 300, 300, 300]
    assert len(recovery_run.calls("exec")) == 8
    assert (
        result["metadata"]["error_details"]["provider_retry_stop_reason"]
        == "wait_budget_exhausted"
    )


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
def test_permanent_failures_do_not_resume(
    recovery_run: RecoveryRun, status: int
) -> None:
    result = recovery_run.run([failure(status=status)])
    assert recovery_run.waits == []
    assert len(recovery_run.calls("exec")) == 1
    assert result["metadata"]["error_details"]["api_error_status"] == status


@pytest.mark.parametrize("status", [408, 500, 502, 503, 504, 529])
def test_transient_provider_failures_resume(
    recovery_run: RecoveryRun, status: int
) -> None:
    result = recovery_run.run([failure(status=status), SUCCESS])
    recovery_run.assert_completed(result)
    assert len(recovery_run.calls("exec")) == 2


def test_original_run_deadline_limits_cooldown(recovery_run: RecoveryRun) -> None:
    result = recovery_run.run([failure()] * 3, timeout=180)
    assert recovery_run.waits == [60]
    assert len(recovery_run.calls("exec")) == 2
    assert (
        result["metadata"]["error_details"]["provider_retry_stop_reason"]
        == "run_deadline"
    )


def test_successful_compaction_exit_resets_backoff(recovery_run: RecoveryRun) -> None:
    compacted = {"events": [pi_message("length")]}
    result = recovery_run.run([failure(), compacted, failure(), SUCCESS])
    recovery_run.assert_completed(result)
    assert recovery_run.waits == [60, 60]
    assert len(recovery_run.calls("exec")) == 4


@pytest.mark.parametrize("wait_budget, expected_waits", [(1000, [900]), (600, [])])
def test_long_provider_hint_is_honored_or_rejected(
    recovery_run: RecoveryRun, wait_budget: float, expected_waits: list[float]
) -> None:
    attempt = failure()
    attempt["events"][-1] = pi_message(
        "error", error='429: {"code":429,"metadata":{"retry_after_seconds":900}}'
    )
    result = recovery_run.run([attempt, SUCCESS], wait_budget=wait_budget)
    assert recovery_run.waits == expected_waits
    assert len(recovery_run.calls("exec")) == 1 + len(expected_waits)
    if expected_waits:
        recovery_run.assert_completed(result)
    else:
        assert result["metadata"]["error_details"]["retry_after_seconds"] == 900


def test_overslept_cooldown_does_not_extend_run_deadline(
    recovery_run: RecoveryRun,
) -> None:
    recovery_run.extra_wait = 200
    result = recovery_run.run([failure()], timeout=180)
    assert len(recovery_run.calls("exec")) == 1
    details = result["metadata"]["error_details"]
    assert details["timed_out"]
    assert details["api_error_status"] == 429
    assert details["provider_retry_wait_seconds"] == 260


@pytest.mark.parametrize("stops_during_wait", [False, True])
def test_stopped_sandbox_is_not_recreated(
    recovery_run: RecoveryRun, monkeypatch: pytest.MonkeyPatch, stops_during_wait: bool
) -> None:
    monkeypatch.setattr(
        _cli_runner,
        "is_docker_container_running",
        lambda _name: stops_during_wait and recovery_run.elapsed == 0,
    )
    result = recovery_run.run([failure()])
    assert recovery_run.waits == ([60] if stops_during_wait else [])
    assert len(recovery_run.calls("exec")) == 1
    assert len(recovery_run.calls("create")) == 1
    assert (
        result["metadata"]["error_details"]["provider_retry_stop_reason"]
        == "sandbox_stopped"
    )


@pytest.mark.parametrize("stop_reason", ["error", "aborted"])
def test_error_and_aborted_messages_do_not_reset_backoff(
    recovery_run: RecoveryRun, stop_reason: str
) -> None:
    second = failure()
    second["events"].insert(1, pi_message(stop_reason))
    result = recovery_run.run([failure(), second, SUCCESS])
    recovery_run.assert_completed(result)
    assert recovery_run.waits == [60, 120]


def test_zero_wait_budget_disables_provider_resumes(recovery_run: RecoveryRun) -> None:
    result = recovery_run.run([failure()], wait_budget=0)
    assert recovery_run.waits == []
    assert len(recovery_run.calls("exec")) == 1
    assert result["metadata"]["error_details"]["provider_retry_count"] == 0


def test_missing_session_never_restarts_task(recovery_run: RecoveryRun) -> None:
    attempt = {"events": [pi_message("error", error="429: Too Many Requests")]}
    result = recovery_run.run([attempt])
    assert recovery_run.waits == []
    assert len(recovery_run.calls("exec")) == 1
    assert (
        result["metadata"]["error_details"]["provider_retry_stop_reason"]
        == "missing_session"
    )


def test_unrelated_error_after_recovery_has_no_stale_provider_failure(
    recovery_run: RecoveryRun,
) -> None:
    unrelated = {"events": [pi_message("error", error="Invalid model configuration")]}
    result = recovery_run.run([failure(), unrelated])
    assert recovery_run.waits == [60]
    assert len(recovery_run.calls("exec")) == 2
    assert "api_error_status" not in result["metadata"].get("error_details", {})


def test_claude_uses_same_session_and_wait_budget(recovery_run: RecoveryRun) -> None:
    attempt = {
        "events": [
            {
                "type": "result",
                "terminal_reason": "api_error",
                "api_error_status": 429,
                "session_id": "session-1",
            }
        ]
    }
    result = recovery_run.run(
        [attempt, {"finish": True, "events": []}], agent="claudecode"
    )
    recovery_run.assert_completed(result)
    assert recovery_run.waits == [60]
    assert len(recovery_run.calls("create")) == 1
    command = recovery_run.calls("exec")[1]
    assert command[command.index("--resume") + 1] == "session-1"


@pytest.mark.parametrize("budget", [-1.0, float("inf"), float("nan"), True])
def test_invalid_wait_budget_is_rejected_before_creating_sandbox(
    tmp_path: Path, budget: float
) -> None:
    with pytest.raises(ValueError, match="provider_retry_wait_seconds"):
        run_pi_task("task", tmp_path, provider_retry_wait_seconds=budget)
