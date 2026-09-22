import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from minisweagent import models
from minisweagent.environments.docker import DockerEnvironment

from latch_eval_tools.harness import minisweagent


def _run_miniswe(
    monkeypatch: pytest.MonkeyPatch,
    work_dir: Path,
    commands: list[str | BaseException],
    **kwargs: Any,
) -> tuple[dict[str, Any], Mock]:
    steps = iter(commands)

    def query(_: list[dict[str, Any]]) -> dict[str, Any]:
        command = next(steps)
        if isinstance(command, BaseException):
            raise command
        return {
            "role": "assistant",
            "content": command,
            "extra": {"actions": [{"command": command}]},
        }

    model = Mock()
    model.query.side_effect = query
    model.get_template_vars.return_value = {}
    model.serialize.return_value = {}
    model.format_message.side_effect = lambda **message: message
    model.format_observation_messages.side_effect = lambda _message, outputs, _vars: [
        {"role": "tool", "content": json.dumps(output)} for output in outputs
    ]
    monkeypatch.setattr(models, "get_model", lambda *_, **__: model)
    monkeypatch.setattr(minisweagent, "ensure_docker_image", lambda _: None)
    monkeypatch.setattr(minisweagent, "get_agent_workspace_dir", lambda _: work_dir)
    monkeypatch.setattr(
        minisweagent, "is_docker_container_running", lambda *_, **__: True
    )
    monkeypatch.setattr(
        minisweagent, "is_docker_container_oom_killed", lambda *_, **__: False
    )
    monkeypatch.setattr(
        DockerEnvironment,
        "_start_container",
        lambda self: setattr(self, "container_id", "test"),
    )
    monkeypatch.setattr(DockerEnvironment, "cleanup", lambda _: None)

    def execute(self: Any, action: dict, **_: Any) -> dict[str, Any]:
        process = subprocess.run(
            ["bash", "-c", action["command"]],
            cwd=work_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        output = {
            "output": process.stdout + process.stderr,
            "returncode": process.returncode,
        }
        self._check_finished(output)
        return output

    monkeypatch.setattr(DockerEnvironment, "execute", execute)
    result = minisweagent.run_minisweagent_task(
        "Write the requested report.",
        work_dir,
        model_name="test/model",
        memory_limit_bytes=1024,
        prompt_suffix=None,
        **kwargs,
    )
    return result, model


@pytest.mark.parametrize("completion", [False, True])
@pytest.mark.parametrize("report_path", ["report.md", "nested/custom.md"])
def test_report_draft_and_incidental_finished_do_not_submit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    completion: bool,
    report_path: str,
) -> None:
    result, model = _run_miniswe(
        monkeypatch,
        tmp_path,
        [
            f"mkdir -p nested; printf draft > {report_path}; printf done > finished.txt",
            f"printf final > {report_path}; echo RUN_COMPLETE",
        ],
        completion=completion,
        benchmark=True,
        completion_file_path=report_path,
        system_prompt="Analyze the data carefully.",
    )

    assert model.query.call_count == 2
    assert (tmp_path / report_path).read_text() == "final"
    assert result["answer"] is None
    assert "error_details" not in result["metadata"]
    assert (
        "echo RUN_COMPLETE"
        in result["metadata"]["harness_config"]["agent_config"]["instance_template"]
    )
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["info"]["exit_status"] == "Submitted"


@pytest.mark.parametrize("incidental_file", ["finished.txt", "eval_answer.json"])
def test_submit_marker_requires_the_declared_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, incidental_file: str
) -> None:
    (tmp_path / incidental_file).write_text("{}")
    result, model = _run_miniswe(
        monkeypatch,
        tmp_path,
        ["echo RUN_COMPLETE", "printf final > report.md; echo RUN_COMPLETE"],
        completion_file_path="report.md",
    )

    assert model.query.call_count == 2
    assert "error_details" not in result["metadata"]


def test_failed_submit_command_does_not_end_report_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "report.md").write_text("draft")
    result, model = _run_miniswe(
        monkeypatch,
        tmp_path,
        ["echo RUN_COMPLETE; exit 1", "echo RUN_COMPLETE"],
        completion_file_path="report.md",
    )

    assert model.query.call_count == 2
    assert "error_details" not in result["metadata"]


@pytest.mark.parametrize(
    "failure", [RuntimeError("provider failed"), minisweagent.AgentTimeoutError()]
)
def test_existing_report_does_not_hide_errors_or_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: BaseException
) -> None:
    result, _ = _run_miniswe(
        monkeypatch,
        tmp_path,
        ["printf draft > report.md", failure],
        completion_file_path="report.md",
    )

    assert result["answer"] is None
    error = result["metadata"]["error_details"]
    if isinstance(failure, minisweagent.AgentTimeoutError):
        assert error["timed_out"] is True
        assert error["error"] == "Agent timed out"
    else:
        assert error["timed_out"] is False
        assert "provider failed" in error["error"]


def test_existing_report_does_not_hide_native_limit_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, model = _run_miniswe(
        monkeypatch,
        tmp_path,
        ["printf draft > report.md"],
        completion_file_path="report.md",
        agent_config={"step_limit": 1},
    )

    assert model.query.call_count == 1
    assert "LimitsExceeded" in result["metadata"]["error_details"]["error"]


@pytest.mark.parametrize("completion_file_path", [None, "finished.txt"])
def test_finished_file_completion_preserves_legacy_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, completion_file_path: str | None
) -> None:
    result, model = _run_miniswe(
        monkeypatch,
        tmp_path,
        ["printf finished > finished.txt"],
        completion=True,
        completion_file_path=completion_file_path,
    )

    assert model.query.call_count == 1
    assert result["answer"]["finished_file_contents"] == "finished"
    assert result["answer"]["last_message"] == "printf finished > finished.txt"
    assert "error_details" not in result["metadata"]


def test_default_still_requires_json_answer_and_submit_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, model = _run_miniswe(
        monkeypatch,
        tmp_path,
        [
            "printf draft > report.md; echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            "printf '{}' > eval_answer.json; echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
    )

    assert model.query.call_count == 2
    assert result["answer"] == {}
    assert "error_details" not in result["metadata"]
