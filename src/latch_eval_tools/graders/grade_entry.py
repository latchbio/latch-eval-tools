"""Self-contained entrypoint run INSIDE the ``--network none`` grading container.
"""

import contextlib
import json
import math
import os
import sys

RESULT_SENTINEL = "@@ARTIFACT_GRADE_RESULT@@"


def _strip_code_fences(source: str) -> str:
    """Drop an optional ``` / ```py fence wrapping a code string."""

    text = source.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    return text


def fn_source(source) -> str:
    """Accept a code string or a list of code lines; return source to exec ("" if neither)."""

    if isinstance(source, list):
        if not all(isinstance(line, str) for line in source):
            return ""
        source = "\n".join(source)

    if not isinstance(source, str):
        return ""

    return _strip_code_fences(source)


def compute_values(agent_answer: dict, config: dict) -> tuple[dict, list]:
    """UNSAFE: exec each field's untrusted fn -> {field: number|None} + errors."""

    ground_truth = config.get("ground_truth", {})
    artifact_fns = config.get("artifact_fns", {})
    reference_artifact_dir = config.get("reference_artifact_dir")

    computed: dict = {}
    errors: list = []

    for field in ground_truth:

        spec = artifact_fns[field]
        entrypoint = spec.get("entrypoint") or "grade"
        path = agent_answer.get(field)

        if not isinstance(path, str) or path == "":
            computed[field] = None
            errors.append(f"{field}: answer field missing or not a filename")
            continue

        if not os.path.exists(path):
            computed[field] = None
            errors.append(f"{field}: file not found: {path}")
            continue

        try:
            namespace: dict = {"REFERENCE_ARTIFACT_DIR": reference_artifact_dir}
            exec(fn_source(spec["fn"]), namespace)
            fn = namespace.get(entrypoint)

            if not callable(fn):
                computed[field] = None
                errors.append(f"{field}: grader_fn defines no callable {entrypoint!r}")
                continue

            value = fn(path)

            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                computed[field] = None
                errors.append(f"{field}: grader_fn returned non-numeric {value!r}")
                continue

            computed[field] = float(value)

        except Exception as exc:
            computed[field] = None
            errors.append(f"{field}: grader_fn raised {type(exc).__name__}: {exc}")

    return computed, errors


def main() -> None:

    payload = json.loads(sys.stdin.read())

    # Keep the untrusted fn's prints off stdout so they can't corrupt the result
    # line (they still reach stderr for debugging).
    with contextlib.redirect_stdout(sys.stderr):
        computed, errors = compute_values(payload["agent_answer"], payload["config"])

    sys.stdout.write(
        RESULT_SENTINEL + json.dumps({"computed": computed, "errors": errors}) + "\n"
    )


if __name__ == "__main__":
    main()
