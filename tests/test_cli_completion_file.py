import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from unittest.mock import Mock

import pytest

from latch_eval_tools.harness import _cli_runner, claudecode, grokbuild

AgentType = Literal["claudecode", "grokbuild"]


@pytest.fixture(params=["claudecode", "grokbuild"])
def agent_type(request: pytest.FixtureRequest) -> AgentType:
    return request.param


def _process(agent_type: AgentType, returncode: int | None = 0) -> Mock:
    event = (
        {"type": "result", "session_id": "test-session", "subtype": "success"}
        if agent_type == "claudecode"
        else {"type": "end", "sessionId": "test-session"}
    )
    process = Mock(
        args=[agent_type],
        stdin=io.StringIO(),
        stdout=io.StringIO(json.dumps(event) + "\n"),
        stderr=io.StringIO(),
        returncode=returncode,
    )
    process.poll.return_value = returncode
    return process


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    work_dir: Path,
    agent_type: AgentType,
    popen: Callable[..., Mock],
    *,
    completion: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    monkeypatch.delenv(_cli_runner.FALLBACK_API_KEYS_ENV, raising=False)
    monkeypatch.setattr(_cli_runner, "ensure_docker_image", lambda _: None)
    monkeypatch.setattr(_cli_runner, "get_agent_workspace_dir", lambda _: work_dir)
    monkeypatch.setattr(
        _cli_runner, "_create_cli_container", lambda **_: "/root/.agent"
    )
    monkeypatch.setattr(_cli_runner, "_start_cli_container", lambda _: None)
    monkeypatch.setattr(_cli_runner, "teardown_container", lambda _: None)
    monkeypatch.setattr(_cli_runner, "is_docker_container_running", lambda _: True)
    monkeypatch.setattr(_cli_runner, "is_docker_container_oom_killed", lambda _: False)
    monkeypatch.setattr(_cli_runner, "MAX_CLAUDECODE_ANSWER_RESUMES", 1)
    monkeypatch.setattr(_cli_runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(_cli_runner.subprocess, "Popen", popen)
    return _cli_runner._run_cli_agent(
        agent_type=agent_type,
        cli_command=[agent_type],
        task_prompt="Write the requested report.",
        work_dir=work_dir,
        memory_limit_bytes=1024,
        benchmark=True,
        completion=completion,
        **kwargs,
    )


def test_declared_report_completes_after_one_clean_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, agent_type: AgentType
) -> None:
    (tmp_path / "report.md").write_text("# Completed report\n")
    popen = Mock(side_effect=lambda *_, **__: _process(agent_type))

    result = _run_cli(
        monkeypatch, tmp_path, agent_type, popen, completion_file_path="report.md"
    )

    assert popen.call_count == 1
    assert "error_details" not in result["metadata"]
    assert (tmp_path / "report.md").read_text() == "# Completed report\n"


@pytest.mark.parametrize("incidental_file", [None, "finished.txt", "eval_answer.json"])
def test_missing_report_resumes_even_if_legacy_answer_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_type: AgentType,
    incidental_file: str | None,
) -> None:
    if incidental_file is not None:
        (tmp_path / incidental_file).write_text("{}")

    def launch(*_: Any, **__: Any) -> Mock:
        if popen.call_count == 2:
            (tmp_path / "report.md").write_text("# Completed report\n")
        return _process(agent_type)

    popen = Mock(side_effect=launch)
    result = _run_cli(
        monkeypatch, tmp_path, agent_type, popen, completion_file_path="report.md"
    )

    assert popen.call_count == 2
    assert "--resume" in popen.call_args.args[0]
    assert "error_details" not in result["metadata"]
    assert (tmp_path / "report.md").read_text() == "# Completed report\n"


def test_missing_report_after_resumes_is_left_to_contract_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, agent_type: AgentType
) -> None:
    (tmp_path / "finished.txt").write_text("done")
    popen = Mock(side_effect=lambda *_, **__: _process(agent_type))

    result = _run_cli(
        monkeypatch, tmp_path, agent_type, popen, completion_file_path="report.md"
    )

    assert popen.call_count == 2
    assert result["answer"] is None
    assert not (tmp_path / "report.md").exists()
    assert "error_details" not in result["metadata"]


def test_unconfigured_report_preserves_legacy_completion_requirement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, agent_type: AgentType
) -> None:
    (tmp_path / "report.md").write_text("# Unrelated report\n")
    popen = Mock(side_effect=lambda *_, **__: _process(agent_type))

    result = _run_cli(monkeypatch, tmp_path, agent_type, popen, completion=True)

    assert popen.call_count == 2
    assert result["answer"] is None
    assert "finished.txt" in result["metadata"]["error_details"]["error"]


@pytest.mark.parametrize("legacy_file", ["finished.txt", "eval_answer.json"])
def test_legacy_answer_still_completes_without_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_type: AgentType,
    legacy_file: str,
) -> None:
    (tmp_path / legacy_file).write_text("{}")
    popen = Mock(side_effect=lambda *_, **__: _process(agent_type))

    result = _run_cli(
        monkeypatch,
        tmp_path,
        agent_type,
        popen,
        completion=legacy_file == "finished.txt",
    )

    assert popen.call_count == 1
    assert "error_details" not in result["metadata"]
    assert result["answer"] is not None


@pytest.mark.parametrize("completion", [True, False])
def test_declared_finished_marker_preserves_legacy_answer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_type: AgentType,
    completion: bool,
) -> None:
    (tmp_path / "finished.txt").write_text("Task complete\n")
    popen = Mock(side_effect=lambda *_, **__: _process(agent_type))

    result = _run_cli(
        monkeypatch,
        tmp_path,
        agent_type,
        popen,
        completion=completion,
        completion_file_path="finished.txt",
    )

    assert popen.call_count == 1
    assert "error_details" not in result["metadata"]
    assert result["answer"]["finished_file_contents"] == "Task complete\n"


def test_report_created_while_running_does_not_stop_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, agent_type: AgentType
) -> None:
    process = _process(agent_type, returncode=None)

    def poll() -> int | None:
        if process.poll.call_count == 1:
            (tmp_path / "report.md").write_text("# Report in progress\n")
            return None
        (tmp_path / "report.md").write_text("# Completed report\n")
        process.returncode = 0
        return 0

    process.poll.side_effect = poll
    popen = Mock(return_value=process)
    result = _run_cli(
        monkeypatch, tmp_path, agent_type, popen, completion_file_path="report.md"
    )

    assert process.poll.call_count == 2
    process.terminate.assert_not_called()
    process.kill.assert_not_called()
    assert popen.call_count == 1
    assert "error_details" not in result["metadata"]
    assert (tmp_path / "report.md").read_text() == "# Completed report\n"


def test_nonzero_exit_with_report_is_still_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, agent_type: AgentType
) -> None:
    (tmp_path / "report.md").write_text("# Partial report\n")
    popen = Mock(return_value=_process(agent_type, returncode=2))

    result = _run_cli(
        monkeypatch, tmp_path, agent_type, popen, completion_file_path="report.md"
    )

    assert popen.call_count == 1
    assert result["answer"] is None
    assert "exited with code 2" in result["metadata"]["error_details"]["error"]


def test_timeout_with_report_is_still_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, agent_type: AgentType
) -> None:
    (tmp_path / "report.md").write_text("# Partial report\n")
    now = 0.0
    monkeypatch.setattr(_cli_runner.time, "time", lambda: now)
    process = _process(agent_type, returncode=None)

    def poll() -> None:
        nonlocal now
        now = 61.0

    def kill() -> None:
        process.returncode = -9

    process.poll.side_effect = poll
    process.kill.side_effect = kill
    popen = Mock(return_value=process)
    result = _run_cli(
        monkeypatch,
        tmp_path,
        agent_type,
        popen,
        completion_file_path="report.md",
        eval_timeout=60,
    )

    assert popen.call_count == 1
    process.kill.assert_called_once()
    assert result["answer"] is None
    assert result["metadata"]["timed_out"] is True
    assert "timed out" in result["metadata"]["error_details"]["error"]


@pytest.mark.parametrize("status_code", [429, 529])
def test_provider_error_with_report_retries_and_remains_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status_code: int
) -> None:
    (tmp_path / "report.md").write_text("# Partial report\n")
    monkeypatch.setattr(_cli_runner, "PROVIDER_MAX_RESUMES", 1)
    monkeypatch.setattr(_cli_runner, "provider_retry_delay_seconds", lambda *_: 0)

    def launch(*_: Any, **__: Any) -> Mock:
        process = _process("claudecode")
        process.stdout = io.StringIO(
            json.dumps(
                {
                    "type": "result",
                    "session_id": "test-session",
                    "terminal_reason": "api_error",
                    "api_error_status": status_code,
                }
            )
            + "\n"
        )
        return process

    popen = Mock(side_effect=launch)
    result = _run_cli(
        monkeypatch, tmp_path, "claudecode", popen, completion_file_path="report.md"
    )

    assert popen.call_count == 2
    assert result["answer"] is None
    error = result["metadata"]["error_details"]
    assert error["api_error_status"] == status_code
    assert error["provider_retry_count"] == 1


def test_public_wrapper_passes_declared_completion_file_to_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, agent_type: AgentType
) -> None:
    if agent_type == "claudecode":
        module = claudecode
        run = claudecode.run_claudecode_task
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    else:
        module = grokbuild
        run = grokbuild.run_grokbuild_task
        monkeypatch.setenv("XAI_API_KEY", "test-key")
    runner = Mock(return_value={"answer": None, "metadata": {}})
    monkeypatch.setattr(module, "_run_cli_agent", runner)

    result = run(
        "Write the report.",
        tmp_path,
        prompt_suffix="",
        completion=True,
        benchmark=True,
        completion_file_path="reports/report.md",
    )

    assert result is runner.return_value
    assert runner.call_args.kwargs["completion_file_path"] == "reports/report.md"
