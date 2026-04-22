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
    type, non-dict ``config``) produce a failing ``GraderResult`` at that
    index so the output length and ordering always match the input.
    """
    from . import get_grader  # noqa: PLC0415 -- avoid circular import at module load

    results: list[GraderResult] = []

    for index, spec in enumerate(grader_specs):
        label = f"graders[{index}]"

        if not isinstance(spec, dict):
            results.append(
                _invalid_spec_result(
                    agent_answer,
                    f"{label}: spec must be object, got {type(spec).__name__}",
                )
            )
            continue

        grader_type = spec.get("type")
        sub_config = spec.get("config", {})

        if not isinstance(grader_type, str) or not grader_type:
            results.append(
                _invalid_spec_result(
                    agent_answer,
                    f"{label}: missing or invalid 'type'",
                )
            )
            continue

        if not isinstance(sub_config, dict):
            results.append(
                _invalid_spec_result(
                    agent_answer,
                    f"{label} ({grader_type}): 'config' must be object, "
                    f"got {type(sub_config).__name__}",
                )
            )
            continue

        try:
            sub_grader = get_grader(grader_type)
        except ValueError as exc:
            results.append(_invalid_spec_result(agent_answer, f"{label}: {exc}"))
            continue

        results.append(sub_grader.evaluate_answer(agent_answer, sub_config))

    return results


def _invalid_spec_result(agent_answer: dict, message: str) -> GraderResult:
    return GraderResult(
        passed=False,
        metrics={"error": message},
        reasoning=f"Grader spec error: {message}",
        agent_answer=agent_answer,
        score=0.0,
        field_scores={},
    )
