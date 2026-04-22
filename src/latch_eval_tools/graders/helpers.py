from pydantic import ValidationError

from ..types import GraderSpec
from .base import GraderResult


def grade_multiple_graders_single_answer(
    agent_answer: dict, grader_specs: list
) -> list[GraderResult]:
    """Run every grader in ``grader_specs`` against ``agent_answer``.

    ``grader_specs`` is a list of ``{"type": <str>, "config": <dict>}`` entries
    (the same shape used by the top-level ``graders`` field in an eval JSON).
    Each sub-grader receives the full ``agent_answer`` and its own sub-config;
    sub-graders are expected to own disjoint answer fields (enforced at lint
    time via E051).

    Returns a list of :class:`GraderResult` objects aligned 1:1 with
    ``grader_specs``. Malformed specs (non-dict, missing ``type``, unknown
    type, non-dict ``config``) produce a ``GraderResult`` with
    ``passed=None`` and ``score=None`` at that index so the output length
    and ordering always match the input. ``None`` signals a tooling
    misconfiguration that callers should distinguish from a real agent
    pass/fail (``True``/``False``).
    """
    from . import get_grader  # noqa: PLC0415 -- avoid circular import at module load

    results: list[GraderResult] = []

    for index, spec in enumerate(grader_specs):
        label = f"graders[{index}]"

        try:
            parsed = GraderSpec.model_validate(spec)
        except ValidationError as exc:
            results.append(
                _invalid_spec_result(
                    agent_answer, f"{label}: {_format_validation_error(exc)}"
                )
            )
            continue

        try:
            sub_grader = get_grader(parsed.type)
        except ValueError as exc:
            results.append(_invalid_spec_result(agent_answer, f"{label}: {exc}"))
            continue

        results.append(sub_grader.evaluate_answer(agent_answer, parsed.config))

    return results


def _invalid_spec_result(agent_answer: dict, message: str) -> GraderResult:
    return GraderResult(
        passed=None,
        metrics={"error": message},
        reasoning=f"Grader spec error: {message}",
        agent_answer=agent_answer,
        score=None,
        field_scores={},
    )


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {err.get('msg', '')}")
    return "; ".join(parts)
