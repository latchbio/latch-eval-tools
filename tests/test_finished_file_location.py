from pathlib import Path

from latch_eval_tools.harness.utils import find_finished_file


def test_finds_marker_at_workspace_root(tmp_path: Path) -> None:
    (tmp_path / "finished.txt").write_text("finished")
    assert find_finished_file(tmp_path) == tmp_path / "finished.txt"


def test_returns_none_when_no_marker(tmp_path: Path) -> None:
    assert find_finished_file(tmp_path) is None


def test_finds_marker_written_in_a_subdirectory(tmp_path: Path) -> None:
    # Agents working inside a project subdir sometimes write finished.txt
    # relative to their cwd rather than at the workspace root.
    sub = tmp_path / "repo"
    sub.mkdir()
    (sub / "finished.txt").write_text("finished")
    assert find_finished_file(tmp_path) == sub / "finished.txt"


def test_root_marker_takes_precedence_over_subdirectory(tmp_path: Path) -> None:
    (tmp_path / "finished.txt").write_text("root")
    sub = tmp_path / "repo"
    sub.mkdir()
    (sub / "finished.txt").write_text("subdir")
    assert find_finished_file(tmp_path) == tmp_path / "finished.txt"


def test_ignores_marker_under_read_only_data_mount(tmp_path: Path) -> None:
    # `data` is the read-only dataset mount; a file of the same name there
    # must not be mistaken for the agent's completion marker.
    data = tmp_path / "data"
    data.mkdir()
    (data / "finished.txt").write_text("dataset-decoy")
    assert find_finished_file(tmp_path) is None


def test_prefers_real_subdir_marker_over_data_decoy(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "finished.txt").write_text("dataset-decoy")
    sub = tmp_path / "repo"
    sub.mkdir()
    (sub / "finished.txt").write_text("finished")
    assert find_finished_file(tmp_path) == sub / "finished.txt"
