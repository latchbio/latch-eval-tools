import traceback
from typing import Any

from pydantic import ValidationError

from ..types import GraderSpec
from .base import GraderResult


def grade_answer_with_specs(
    agent_answer: dict,
    grader_specs: list,
) -> GraderResult | None:
    try:
        per_grader_results = grade_multiple_graders_single_answer(
            agent_answer,
            grader_specs,
        )
    except Exception as exc:
        return GraderResult(
            passed=False,
            metrics={"grader_error": str(exc)},
            reasoning=(
                "Grader failed while evaluating agent output: "
                f"{exc}\n\n{traceback.format_exc()}"
            ),
            agent_answer=agent_answer,
            score=0.0,
            field_scores={},
        )

    if len(per_grader_results) == 0:
        return None
    if len(per_grader_results) == 1 and per_grader_results[0] is not None:
        return per_grader_results[0]

    results: list[GraderResult] = []
    for result in per_grader_results:
        if result is None:
            return GraderResult(
                passed=False,
                metrics={},
                reasoning="Grader spec is malformed or uses an unknown grader type.",
                agent_answer=agent_answer,
                score=0.0,
                field_scores={},
            )
        results.append(result)

    metrics: dict[str, Any] = {}
    field_scores: dict[str, float] = {}
    reasoning_sections: list[str] = []
    scores: list[float] = []
    all_passed = True

    for i, result in enumerate(results):
        prefix = f"graders[{i}]"

        all_passed = all_passed and result.passed
        scores.append(result.score)
        for metric_key, metric_value in result.metrics.items():
            metrics[f"{prefix}.{metric_key}"] = metric_value
        for field_name, field_score in result.field_scores.items():
            field_scores[f"{prefix}.{field_name}"] = field_score

        status = "PASS" if result.passed else "FAIL"
        reasoning_sections.append(
            f"[{prefix}] {status} (score={result.score:.4f})\n{result.reasoning}"
        )

    return GraderResult(
        passed=all_passed,
        metrics=metrics,
        reasoning="\n\n".join(reasoning_sections),
        agent_answer=agent_answer,
        score=sum(scores) / len(scores),
        field_scores=field_scores,
    )


def grade_multiple_graders_single_answer(
    agent_answer: dict,
    grader_specs: list,
) -> list[GraderResult | None]:
    """Run every grader in ``grader_specs`` against ``agent_answer``.

    ``grader_specs`` is a list of ``{"type": <str>, "config": <dict>}`` entries
    (the same shape used by the top-level ``graders`` field in an eval JSON).
    Each sub-grader receives the full ``agent_answer`` and its own sub-config;
    sub-graders are expected to own disjoint answer fields.

    Returns a list aligned 1:1 with ``grader_specs``. A valid spec yields a
    :class:`GraderResult`; any malformed spec (non-dict, missing ``type``,
    unknown type, non-dict ``config``) yields ``None`` at that index so
    callers can distinguish tooling misconfiguration from a real agent
    pass/fail.
    """
    from . import get_grader  # noqa: PLC0415 -- avoid circular import at module load

    per_grader_results: list[GraderResult | None] = []

    for spec in grader_specs:
        try:
            parsed = GraderSpec.model_validate(spec)
        except ValidationError:
            per_grader_results.append(None)
            continue

        try:
            sub_grader = get_grader(parsed.type)
        except ValueError:
            per_grader_results.append(None)
            continue

        per_grader_results.append(
            sub_grader.evaluate_answer(agent_answer, parsed.config)
        )

    return per_grader_results
