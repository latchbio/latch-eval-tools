from __future__ import annotations

import json
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

from latch_eval_tools import EvalRunner, GraderResult

CASES_ROOT = Path(__file__).parent / "fixtures" / "visual_grader_cases"
CASE_DIRS = sorted(path.parent for path in CASES_ROOT.glob("*/task.json"))


@pytest.mark.parametrize("case_dir", CASE_DIRS, ids=lambda path: path.name)
def test_visual_grader_json_case(case_dir: Path, tmp_path: Path) -> None:
    assert {path.name for path in case_dir.iterdir()} == {
        "task.json",
        "submission.json",
    }

    task = json.loads((case_dir / "task.json").read_text())
    submission = json.loads((case_dir / "submission.json").read_text())
    expected = task["metadata"]["expected"]

    runner = EvalRunner(
        case_dir / "task.json",
        workspace_name=str(tmp_path / "workspace"),
        cache_name=str(tmp_path / "cache"),
        benchmark_name="Visual grader JSON smoke test",
    )
    run_result = runner.run(
        agent_function=lambda _task, _work_dir: submission,
    )

    result = run_result["grader_result"]
    assert isinstance(result, GraderResult)
    assert not result.metrics.get("configuration_error")
    assert not result.metrics.get("invalid_answer")
    assert not result.metrics.get("grader_system_error")
    assert result.passed is expected["passed"]
    assert result.score == pytest.approx(expected["score"])

    if "matched_pairs" in expected:
        matched_pairs = [
            [match["reference_index"], match["submitted_index"]]
            for match in result.metrics["matches"]
        ]
        assert matched_pairs == expected["matched_pairs"]
        assert (
            result.metrics["unmatched_reference_indices"]
            == expected["unmatched_reference_indices"]
        )
        assert (
            result.metrics["unmatched_submitted_indices"]
            == expected["unmatched_submitted_indices"]
        )
