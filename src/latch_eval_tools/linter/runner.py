import json
from pathlib import Path

from .schema import LintResult, LintIssue
from .validators import ALL_VALIDATORS


def lint_eval(path: str | Path) -> LintResult:
    path = Path(path)
    result = LintResult(file_path=str(path))

    if not path.exists():
        result.issues.append(LintIssue("error", "E000", f"File not found: {path}"))
        return result

    if not path.suffix == ".json":
        result.issues.append(LintIssue("warning", "W000", f"File does not have .json extension: {path}"))

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result.issues.append(LintIssue("error", "E001", f"Invalid JSON: {e}"))
        return result

    if not isinstance(data, dict):
        result.issues.append(LintIssue("error", "E002", f"Root must be object, got {type(data).__name__}"))
        return result

    for validator in ALL_VALIDATORS:
        result.issues.extend(validator(data))

    return result


def lint_directory(path: str | Path, pattern: str = "**/*.json") -> list[LintResult]:
    path = Path(path)
    results = []

    if not path.exists():
        return [LintResult(
            file_path=str(path),
            issues=[LintIssue("error", "E000", f"Directory not found: {path}")]
        )]

    if not path.is_dir():
        return [lint_eval(path)]

    for json_file in sorted(path.glob(pattern)):
        if json_file.name.startswith("."):
            continue
        results.append(lint_eval(json_file))

    return results


def format_results(results: list[LintResult], format: str = "console") -> str:
    if format == "console":
        return _format_console(results)
    elif format == "json":
        return _format_json(results)
    elif format == "markdown":
        return _format_markdown(results)
    else:
        raise ValueError(f"Unknown format: {format}")


def _format_console(results: list[LintResult]) -> str:
    lines = []
    total_errors = 0
    total_warnings = 0

    for result in results:
        if not result.issues:
            continue

        lines.append(f"\n{result.file_path}")
        for issue in result.issues:
            prefix = "  ✗" if issue.level == "error" else "  ⚠"
            lines.append(f"{prefix} {issue}")

        total_errors += result.error_count
        total_warnings += result.warning_count

    lines.append(f"\n{'='*50}")
    lines.append(f"Files checked: {len(results)}")
    lines.append(f"Files with issues: {sum(1 for r in results if r.issues)}")
    lines.append(f"Errors: {total_errors}, Warnings: {total_warnings}")

    passed = sum(1 for r in results if r.passed)
    lines.append(f"Passed: {passed}/{len(results)}")

    return "\n".join(lines)


def _format_json(results: list[LintResult]) -> str:
    output = {
        "summary": {
            "files_checked": len(results),
            "files_with_issues": sum(1 for r in results if r.issues),
            "total_errors": sum(r.error_count for r in results),
            "total_warnings": sum(r.warning_count for r in results),
            "passed": sum(1 for r in results if r.passed),
        },
        "results": [
            {
                "file": r.file_path,
                "passed": r.passed,
                "issues": [
                    {"level": i.level, "code": i.code, "message": i.message, "location": i.location}
                    for i in r.issues
                ]
            }
            for r in results
        ]
    }
    return json.dumps(output, indent=2)


def _format_markdown(results: list[LintResult]) -> str:
    lines = ["# Lint Results\n"]

    total_errors = sum(r.error_count for r in results)
    total_warnings = sum(r.warning_count for r in results)
    passed = sum(1 for r in results if r.passed)

    lines.append(f"**Files checked:** {len(results)}")
    lines.append(f"**Passed:** {passed}/{len(results)}")
    lines.append(f"**Errors:** {total_errors}, **Warnings:** {total_warnings}\n")

    files_with_issues = [r for r in results if r.issues]
    if not files_with_issues:
        lines.append("All files passed validation.")
        return "\n".join(lines)

    lines.append("## Issues\n")
    for result in files_with_issues:
        lines.append(f"### `{result.file_path}`\n")
        lines.append("| Level | Code | Message | Location |")
        lines.append("|-------|------|---------|----------|")
        for issue in result.issues:
            loc = issue.location or "-"
            lines.append(f"| {issue.level} | {issue.code} | {issue.message} | {loc} |")
        lines.append("")

    return "\n".join(lines)
