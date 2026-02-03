#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from latch_eval_tools.linter import lint_eval, lint_directory, LintResult
from latch_eval_tools.linter.explanations import get_explanation


def format_console_rich(results: list[LintResult]) -> str:
    lines = []
    total_errors = 0
    total_warnings = 0

    for result in results:
        lines.append(f"\nChecking: {result.file_path}")
        lines.append("─" * 50)

        if not result.issues:
            lines.append("✓ All checks passed")
            continue

        for issue in result.issues:
            prefix = "✗" if issue.level == "error" else "⚠"
            explanation = get_explanation(issue.code)

            lines.append(f"\n{prefix} {issue.code}: {issue.message}")

            if explanation:
                lines.append(f"")
                lines.append(f"  Fix: {explanation.example_before} → {explanation.example_after}")
                if explanation.doc_link:
                    lines.append(f"  Docs: {explanation.doc_link}")

            if issue.location:
                lines.append(f"  Location: {issue.location}")

        total_errors += result.error_count
        total_warnings += result.warning_count

    lines.append("")
    lines.append("─" * 50)
    lines.append(f"Result: {total_errors} error(s), {total_warnings} warning(s)")
    lines.append(f"Files: {sum(1 for r in results if r.passed)}/{len(results)} passed")

    return "\n".join(lines)


def format_json_output(results: list[LintResult]) -> str:
    output = {
        "summary": {
            "files_checked": len(results),
            "files_passed": sum(1 for r in results if r.passed),
            "total_errors": sum(r.error_count for r in results),
            "total_warnings": sum(r.warning_count for r in results),
        },
        "results": [],
    }

    for result in results:
        result_entry = {
            "file": result.file_path,
            "passed": result.passed,
            "errors": result.error_count,
            "warnings": result.warning_count,
            "issues": [],
        }

        for issue in result.issues:
            issue_entry: dict = {
                "level": issue.level,
                "code": issue.code,
                "message": issue.message,
            }
            if issue.location:
                issue_entry["location"] = issue.location

            explanation = get_explanation(issue.code)
            if explanation:
                issue_entry["fix"] = {
                    "before": explanation.example_before,
                    "after": explanation.example_after,
                }
                if explanation.doc_link:
                    issue_entry["docs"] = explanation.doc_link

            result_entry["issues"].append(issue_entry)

        output["results"].append(result_entry)

    return json.dumps(output, indent=2)


VALID_CATEGORIES = ["qc", "normalization", "dimensionality_reduction", "clustering", "cell_typing", "differential_expression", "spatial_analysis"]


def main():
    parser = argparse.ArgumentParser(
        prog="eval-lint",
        description="Validate eval JSON files locally (no credentials required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  eval-lint path/to/eval.json           # Lint single file
  eval-lint evals/my_dataset/           # Lint directory
  eval-lint evals/ --category qc        # Lint only QC evals
  eval-lint evals/ --format json        # JSON output for CI/CD
  eval-lint evals/ -q                   # Quiet mode (exit code only)

Exit codes:
  0  All files passed validation
  1  One or more files have errors
""",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to eval JSON file or directory containing eval files",
    )
    parser.add_argument(
        "--category", "-c",
        choices=VALID_CATEGORIES,
        help="Only lint evals with this metadata.task category",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["console", "json"],
        default="console",
        help="Output format (default: console)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode: only show summary and exit code",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.json",
        help="Glob pattern for finding files in directory (default: **/*.json)",
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    if args.path.is_file():
        results = [lint_eval(args.path)]
    else:
        results = lint_directory(args.path, args.pattern)

    if args.category:
        filtered_results = []
        for result in results:
            try:
                with open(result.file_path) as f:
                    eval_data = json.load(f)
                if eval_data.get("metadata", {}).get("task") == args.category:
                    filtered_results.append(result)
            except (json.JSONDecodeError, IOError):
                filtered_results.append(result)
        results = filtered_results

    if not results:
        print("No eval files found", file=sys.stderr)
        sys.exit(1)

    total_errors = sum(r.error_count for r in results)
    all_passed = all(r.passed for r in results)

    if args.quiet:
        passed = sum(1 for r in results if r.passed)
        print(f"{passed}/{len(results)} files passed, {total_errors} error(s)")
    elif args.format == "json":
        print(format_json_output(results))
    else:
        print(format_console_rich(results))

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
