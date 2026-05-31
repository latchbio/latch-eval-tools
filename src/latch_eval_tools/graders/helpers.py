import traceback
from typing import Any

from pydantic import ValidationError

from ..types import GraderSpec
from .base import GraderResult


def grade_answer_with_specs(
    agent_answer: dict,
    grader_specs: list,
) -> tuple[list[GraderResult | None], GraderResult | None]:
    per_grader_results = grade_multiple_graders_single_answer(agent_answer, grader_specs)

    if len(per_grader_results) == 0:
        return per_grader_results, None
    if len(per_grader_results) == 1 and per_grader_results[0] is not None:
        return per_grader_results, per_grader_results[0]

    metrics: dict[str, Any] = {"n_graders": len(per_grader_results)}
    field_scores: dict[str, float] = {}
    reasoning_sections: list[str] = []
    scores: list[float] = []
    all_passed = True
    misconfigured = False

    for i, result in enumerate(per_grader_results):
        spec = grader_specs[i] if i < len(grader_specs) else {}
        grader_type = (
            spec.get("type", "unknown") if isinstance(spec, dict) else "unknown"
        )
        prefix = f"graders[{i}]"

        metrics[f"{prefix}.type"] = grader_type

        if result is None:
            all_passed = False
            misconfigured = True
            metrics[f"{prefix}.misconfigured"] = True
            metrics["grader_misconfigured"] = True
            reasoning_sections.append(
                f"[{prefix} type={grader_type}] FAIL\n"
                "Grader spec is malformed or uses an unknown grader type."
            )
            continue

        all_passed = all_passed and result.passed
        scores.append(result.score)
        metrics[f"{prefix}.passed"] = result.passed
        metrics[f"{prefix}.score"] = result.score
        for metric_key, metric_value in result.metrics.items():
            metrics[f"{prefix}.{metric_key}"] = metric_value
        for field_name, field_score in result.field_scores.items():
            field_scores[f"{prefix}.{field_name}"] = field_score

        status = "PASS" if result.passed else "FAIL"
        reasoning_sections.append(
            f"[{prefix} type={grader_type}] {status} "
            f"(score={result.score:.4f})\n{result.reasoning}"
        )

    return per_grader_results, GraderResult(
        passed=all_passed,
        metrics=metrics,
        reasoning="\n\n".join(reasoning_sections),
        agent_answer=agent_answer,
        score=0.0 if misconfigured else sum(scores) / len(scores),
        field_scores=field_scores,
    )


def grade_multiple_graders_single_answer(
    agent_answer: dict,
    grader_specs: list,
) -> list[GraderResult | None]:
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

        try:
            per_grader_results.append(
                sub_grader.evaluate_answer(agent_answer, parsed.config)
            )
        except Exception as exc:
            per_grader_results.append(
                GraderResult(
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
            )

    return per_grader_results
