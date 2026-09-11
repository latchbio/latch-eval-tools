import json

from latch_eval_tools.harness.utils import (
    ANSWER_FILENAMES,
    BENCHMARK_ANSWER_FILENAME,
    BENCHMARK_COMPLETION_MARKER,
    COMPLETION_MARKERS,
    LEGACY_ANSWER_FILENAME,
    LEGACY_COMPLETION_MARKER,
    benchmark_convention,
    find_answer_file,
)


def test_completion_markers_accept_both_conventions() -> None:
    assert LEGACY_COMPLETION_MARKER in COMPLETION_MARKERS
    assert BENCHMARK_COMPLETION_MARKER in COMPLETION_MARKERS


def test_find_answer_file_prefers_benchmark_name(tmp_path) -> None:
    assert find_answer_file(tmp_path) is None

    legacy = tmp_path / LEGACY_ANSWER_FILENAME
    legacy.write_text(json.dumps({"answer": 1}))
    assert find_answer_file(tmp_path) == legacy

    benchmark = tmp_path / BENCHMARK_ANSWER_FILENAME
    benchmark.write_text(json.dumps({"answer": 2}))
    assert find_answer_file(tmp_path) == benchmark

    assert ANSWER_FILENAMES == (LEGACY_ANSWER_FILENAME, BENCHMARK_ANSWER_FILENAME) or (
        ANSWER_FILENAMES == (BENCHMARK_ANSWER_FILENAME, LEGACY_ANSWER_FILENAME)
    )


def test_benchmark_convention_rewrites_legacy_names() -> None:
    text = (
        "The calling program looks for results in `/workspace/eval_answer.json`. "
        "The completion signal is a standalone `echo "
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` command."
    )
    rewritten = benchmark_convention(text)
    assert "analysis_output.json" in rewritten
    assert "echo RUN_COMPLETE" in rewritten
    assert "eval_answer.json" not in rewritten
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in rewritten
